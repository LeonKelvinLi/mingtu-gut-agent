"""The orchestrator: citation validation, the repair loop, and the emergency path.

Two properties matter more than anything else here and both are tested with every
model call stubbed out:

  * A fabricated source number is caught before the answer ships.
  * An honest "我手上的资料没有覆盖" is not treated as a defect. If abstention
    trips the validator, the repair loop pressures the model into inventing a
    citation — the validator would be manufacturing the exact failure it exists
    to prevent.
"""

import pytest

from agent import core
from agent.core import Agent
from conftest import needs_answering_model

PASSAGES = [
    {"id": "p1", "source": "01-strains", "heading": "2.1 LGG", "text": "【2.1 LGG】\n证据等级：A。",
     "evidence_grade": "A", "authority": "meta-analysis", "population": ["儿童"],
     "year": 2024, "is_regulatory": False, "_score": 3.0,
     "citations": ["https://a/1", "https://a/2", "https://a/3", "https://a/4", "https://a/5"]},
    {"id": "p2", "source": "02-diet", "heading": "4.1 膳食纤维", "text": "【4.1 膳食纤维】\n证据等级：B。",
     "evidence_grade": "B", "authority": "rct", "population": ["成人"],
     "year": 2022, "is_regulatory": False, "_score": 2.0, "citations": []},
    {"id": "p3", "source": "03-china-reg", "heading": "4.2 广告法", "text": "【4.2 广告法】\n法规原文。",
     "evidence_grade": None, "authority": "guideline", "population": ["通用"],
     "year": 2023, "is_regulatory": True, "_score": 1.0, "citations": []},
]

NORMAL_ROUTING = {
    "risk": "normal", "category": "菌株证据", "populations": [], "regulatory": False,
    "needs_current_info": False, "sub_queries": ["益生菌 证据"], "red_flags": [],
    "safety_instruction": None, "_fallback": False,
}


# --- validate_citations -------------------------------------------------------

def test_valid_citations_pass():
    answer = "有一项荟萃分析支持该菌株用于儿童腹泻[1]，每日剂量100亿CFU[1][2]。"
    assert Agent.validate_citations(answer, PASSAGES) == []


@pytest.mark.parametrize("answer,bad", [
    ("研究显示该菌株有效[7]。", 7),
    ("法规规定见[10]。", 10),
    ("参考[0]号资料。", 0),
    ("剂量见[2]，机制见[9]。", 9),
])
def test_invented_citation_numbers_are_caught(answer, bad):
    problems = Agent.validate_citations(answer, PASSAGES)
    assert problems, f"编造的编号 [{bad}] 没有被发现"
    assert str(bad) in problems[0]
    assert "1-3" in problems[0], "纠正信息要告诉模型本轮的合法编号范围"


def test_material_claim_without_any_citation_is_caught():
    answer = "这个菌株每日剂量是100亿CFU，有荟萃分析支持它对儿童腹泻有效。"
    problems = Agent.validate_citations(answer, PASSAGES)
    assert problems == ["回答里有实质性结论但完全没有标注来源编号"]


ABSTENTIONS = [
    "关于这个问题，我手上的资料没有覆盖，建议你直接问消化科医生。",
    "现有证据不足，我没法给你一个可靠的剂量建议。",
    "这个我没有查到可靠依据，学界目前也没有定论。",
    "这个说法目前证据不一致，我无法回答哪个结论成立。",
]


@pytest.mark.parametrize("answer", ABSTENTIONS)
def test_honest_abstention_is_not_a_problem(answer):
    """弃答是正确行为，不能被校验器当成缺陷倒逼模型编引用。"""
    assert Agent.validate_citations(answer, PASSAGES) == []


def test_no_passages_means_no_citation_requirement():
    answer = "这一轮没有检索到资料，所以我不能给你结论，建议咨询消化科医生。"
    assert Agent.validate_citations(answer, []) == []


def test_small_talk_without_citations_is_fine():
    answer = "你好，我可以帮你解释检查项目大概是做什么的。"
    assert Agent.validate_citations(answer, PASSAGES) == []


def test_both_problems_are_reported_together():
    answer = "该菌株每日剂量100亿CFU有效[9]。"
    problems = Agent.validate_citations(answer, PASSAGES)
    assert len(problems) == 1 and "9" in problems[0]


# --- _sources -----------------------------------------------------------------

def test_sources_only_reports_what_was_cited():
    sources = Agent._sources("见[2]和[3]。", PASSAGES)
    assert [s["n"] for s in sources] == [2, 3]
    assert sources[0]["heading"] == "4.1 膳食纤维"
    assert sources[1]["source"] == "03-china-reg"
    assert set(sources[0]) == {"n", "heading", "source", "grade", "is_live", "citations"}


def test_sources_drops_out_of_range_numbers():
    assert [s["n"] for s in Agent._sources("见[1]和[9]。", PASSAGES)] == [1]


def test_sources_deduplicates_and_sorts():
    assert [s["n"] for s in Agent._sources("[3][1][3][1]", PASSAGES)] == [1, 3]


def test_sources_caps_the_url_list():
    sources = Agent._sources("见[1]。", PASSAGES)
    assert len(sources[0]["citations"]) == 4


def test_sources_carries_the_evidence_grade():
    sources = Agent._sources("见[1]和[3]。", PASSAGES)
    assert sources[0]["grade"] == "A"
    assert sources[1]["grade"] is None


def test_sources_is_empty_when_nothing_is_cited():
    assert Agent._sources("没有任何引用的回答。", PASSAGES) == []


# --- pipeline harness ---------------------------------------------------------

class StubRetriever:
    def __init__(self, passages):
        self.passages = passages

    def search(self, query, filters=None, top_n=8):
        return [dict(p) for p in self.passages]


@pytest.fixture
def build_agent(monkeypatch):
    """Build an Agent with every model call and the index stubbed out."""
    def build(outputs, routing=None, passages=None, available=True):
        passages = PASSAGES if passages is None else passages
        routing = dict(routing or NORMAL_ROUTING)
        scripted = list(outputs)
        sent = []

        monkeypatch.setattr(core, "Retriever", lambda: StubRetriever(passages))
        monkeypatch.setattr(core, "classify", lambda query: dict(routing))
        monkeypatch.setattr(core, "rerank", lambda q, cands, top_n=8: list(cands)[:top_n])
        monkeypatch.setattr(core, "needs_live", lambda q, best, **kw: False)
        monkeypatch.setattr(core, "search_live", lambda q: [])
        monkeypatch.setattr(core, "available", lambda: available)
        monkeypatch.setattr(core, "provider_label", lambda: "stub-provider")

        def fake_generate(system, messages, max_tokens=8000, effort="high"):
            sent.append([dict(m) for m in messages])
            return scripted.pop(0) if scripted else None

        monkeypatch.setattr(core, "generate", fake_generate)
        return Agent(), sent
    return build


# --- emergency short-circuit (design constraint 1 + 2) ------------------------

def test_emergency_never_reaches_the_model(build_agent):
    routing = dict(NORMAL_ROUTING, risk="emergency", red_flags=["便血", "体重下降"])
    agent, sent = build_agent(outputs=["这段模型输出绝不该出现"], routing=routing)

    result = agent.answer("我最近大便带血，还瘦了十斤")

    assert sent == [], "急症回答必须绕开模型"
    assert result["escalated"] is True
    assert "便血" in result["answer"] and "体重下降" in result["answer"]
    assert "消化内科" in result["answer"]
    assert result["passages"] == []


def test_emergency_still_answers_with_every_provider_down(build_agent):
    """The failure mode we cannot accept is a red flag going quiet on an outage."""
    routing = dict(NORMAL_ROUTING, risk="emergency", red_flags=["呕血"])
    agent, sent = build_agent(outputs=[], routing=routing, available=False)

    result = agent.answer("我吐血了")

    assert result["escalated"] is True
    assert "急诊" in result["answer"]
    assert "error" not in result
    assert sent == []


def test_no_model_key_still_reports_working_retrieval(build_agent):
    agent, sent = build_agent(outputs=[], available=False)

    result = agent.answer("益生菌要吃多久")

    assert result["error"] == "no_model_key"
    assert result["passages"], "没有模型时检索层仍然要工作"
    assert "安全层" in result["answer"]
    assert sent == []


# --- repair loop --------------------------------------------------------------

def test_invented_citation_triggers_one_repair_then_ships(build_agent):
    agent, sent = build_agent(outputs=[
        "该菌株对儿童腹泻有效[9]。",
        "该菌株对儿童腹泻有效，有荟萃分析支持[1]。",
    ])

    result = agent.answer("LGG对儿童腹泻有用吗")

    assert result["repairs"] == 1
    assert result["answer"] == "该菌株对儿童腹泻有效，有荟萃分析支持[1]。"
    assert [s["n"] for s in result["sources"]] == [1]
    assert "error" not in result

    repair_turn = sent[1]
    assert repair_turn[-2]["role"] == "assistant"
    assert repair_turn[-1]["role"] == "system"
    assert "引用" in repair_turn[-1]["content"]
    assert "9" in repair_turn[-1]["content"]


def test_banned_claim_triggers_a_compliance_repair(build_agent):
    agent, sent = build_agent(outputs=[
        "这个菌株可以治疗肠易激综合征[1]。",
        "有人体试验显示这个菌株有助于缓解腹胀[1]。",
    ])

    result = agent.answer("这个菌株能干嘛")

    assert result["repairs"] == 1
    assert "治疗" not in result["answer"]
    assert "合规指令" in sent[1][-1]["content"]
    assert "治疗" in sent[1][-1]["content"]


def test_a_failing_answer_is_never_shipped(build_agent):
    bad = "这个产品可以根治肠易激，100%有效[9]。"
    agent, sent = build_agent(outputs=[bad, bad, bad])

    result = agent.answer("这个产品能根治吗")

    assert len(sent) == core.MAX_REPAIR_ROUNDS + 1
    assert result["error"] == "validation_failed"
    assert bad not in result["answer"]
    assert "咨询消化科医生" in result["answer"]
    assert result["problems"]


def test_generation_failure_degrades_without_raising(build_agent):
    agent, _ = build_agent(outputs=[])

    result = agent.answer("益生菌要吃多久")

    assert result["error"] == "generation_failed"
    assert result["answer"]


def test_clean_answer_needs_no_repair(build_agent):
    agent, sent = build_agent(outputs=["每天25到30克膳食纤维，有随机对照试验支持[2]。"])

    result = agent.answer("每天吃多少纤维")

    assert result["repairs"] == 0
    assert len(sent) == 1
    assert [s["n"] for s in result["sources"]] == [2]
    assert result["provider"] == "stub-provider"


# --- what actually reaches the model -----------------------------------------

def test_message_roles_stay_within_the_supported_set(build_agent):
    routing = dict(NORMAL_ROUTING, risk="caution", red_flags=["便血"],
                   safety_instruction="【安全指令】用户消息中出现了消化道报警症状（便血）。")
    agent, sent = build_agent(outputs=["有效[9]。", "有效[1]。"], routing=routing)
    agent.answer("便血是什么原因")

    for messages in sent:
        assert messages[0]["role"] == "user"
        for message in messages:
            assert message["role"] in {"user", "assistant", "system"}
            assert isinstance(message["content"], str) and message["content"]


def test_safety_instruction_reaches_the_model(build_agent):
    instruction = "【安全指令】用户消息中出现了消化道报警症状（便血）。"
    routing = dict(NORMAL_ROUTING, risk="caution", red_flags=["便血"],
                   safety_instruction=instruction)
    agent, sent = build_agent(outputs=["建议先就医[1]。"], routing=routing)
    agent.answer("便血是什么原因")

    assert any(instruction in m["content"] for m in sent[0])


def test_evidence_block_is_numbered_for_the_citation_contract(build_agent):
    agent, sent = build_agent(outputs=["见[1]。"])
    agent.answer("益生菌要吃多久")

    first_user_turn = sent[0][0]["content"]
    for n in range(1, len(PASSAGES) + 1):
        assert f"[{n}]" in first_user_turn
    assert "证据等级A" in first_user_turn
    assert "法规" in first_user_turn
    assert "益生菌要吃多久" in first_user_turn


def test_history_is_carried_and_not_mutated(build_agent):
    history = [{"role": "user", "content": "上一轮问题"},
               {"role": "assistant", "content": "上一轮回答"}]
    snapshot = [dict(m) for m in history]
    agent, sent = build_agent(outputs=["见[1]。"])
    agent.answer("接着问", history=history)

    assert history == snapshot, "history 不能被就地修改"
    assert sent[0][0]["content"] == "上一轮问题"


def test_no_passages_still_produces_an_answer(build_agent):
    agent, sent = build_agent(outputs=["我手上的资料没有覆盖这个问题。"], passages=[])

    result = agent.answer("这个冷门菌株有什么证据")

    assert result.get("error") is None
    assert "没有覆盖" in result["answer"]
    assert "【本轮没有检索到相关资料】" in sent[0][0]["content"]


# --- live -------------------------------------------------------------------

@needs_answering_model
def test_live_end_to_end_answer_is_cited_and_compliant():
    from agent import safety

    agent = Agent()
    result = agent.answer("每天该吃多少膳食纤维？")

    assert result["answer"]
    assert "error" not in result, result.get("problems")
    assert Agent.validate_citations(result["answer"], result["passages"]) == []
    assert safety.screen_output(result["answer"]) == []


@needs_answering_model
def test_live_emergency_is_still_the_canned_text():
    agent = Agent()
    result = agent.answer("我这两天大便带血，还瘦了十斤，怎么办")

    assert result["escalated"] is True
    assert "消化内科" in result["answer"]
    assert "急诊" in result["answer"]

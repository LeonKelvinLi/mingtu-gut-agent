"""The deterministic safety layer.

This is the file that has to be right. Everything else in the system degrades
when an API is down; this layer is the thing that is supposed to still work, so
it is tested the way it will actually be used: with the sentences a worried
person types on a phone, not the vocabulary a textbook uses.

Test data convention: every string here is something a real user would plausibly
send. If a phrasing only appears in clinical writing it does not belong in the
must-hit table, because screening it proves nothing about production traffic.
"""

import re

import pytest

from agent import safety
from conftest import known_gap


# --- Gate 1: 报警症状 must fire ------------------------------------------------
# (message, expected label). Colloquial first-person reports, the way they arrive.

MUST_HIT = [
    # 便血 — people say 拉屎/便便/手纸, almost never 便血
    ("最近拉屎带血，是不是痔疮啊", "便血"),
    ("大便上面有血丝，怎么回事", "便血"),
    ("擦屁股的时候手纸上有血", "便血"),
    ("我便便里带点血，要紧吗", "便血"),
    ("便后滴血", "便血"),
    ("上完厕所发现滴血", "便血"),
    ("肛门出血了", "便血"),
    ("昨天拉血了", "便血"),
    ("大便带血两周了", "便血"),
    # 黑便
    ("这两天大便发黑，跟柏油一样", "黑便或柏油样便"),
    ("拉的屎黑黑的，吃了猪血算吗", "黑便或柏油样便"),
    ("解黑便三天", "黑便或柏油样便"),
    # 呕血
    ("我吐血了", "呕血"),
    ("呕吐物里有血", "呕血"),
    # 体重下降
    ("最近没减肥但是瘦了十斤", "体重下降"),
    ("我体重下降了好多", "体重下降"),
    ("半年瘦了20斤", "体重下降"),
    ("体重掉了10斤", "体重下降"),
    ("最近掉秤厉害", "体重下降"),
    ("不知道为什么就瘦了", "体重下降"),
    # 吞咽困难
    ("吃东西咽不下去，噎得慌", "吞咽困难"),
    ("吞咽的时候疼", "吞咽困难"),
    ("食物卡在喉咙下不去", "吞咽困难"),
    # 持续呕吐
    ("吐了一天了，一直吐个不停", "持续呕吐"),
    ("反复呕吐好几天", "持续呕吐"),
    ("一直恶心想吐吐了三天", "持续呕吐"),
    # 贫血
    ("体检说我贫血，血红蛋白低", "贫血"),
    # 腹部包块
    ("肚子上摸到一个硬块", "腹部包块"),
    ("腹部有个肿块", "腹部包块"),
    # 夜间症状
    ("半夜经常肚子疼醒", "夜间症状"),
    ("晚上肚子疼醒", "夜间症状"),
    ("夜间腹泻好几次", "夜间症状"),
    # 黄疸
    ("我眼白发黄，皮肤也黄", "黄疸"),
    # 发热伴腹痛
    ("发烧到38度还肚子疼", "发热伴腹痛"),
    ("肚子疼还发烧了", "发热伴腹痛"),
    # 剧烈腹痛
    ("肚子疼得直不起腰", "剧烈腹痛"),
    ("剧烈腹痛两小时了", "剧烈腹痛"),
    # 脱水
    ("一天没尿了", "脱水"),
    ("尿特别少", "脱水"),
    # 家族史
    ("有肠癌家族史要多久做一次肠镜", "消化道疾病家族史"),
    ("肠癌家族史需要注意什么", "消化道疾病家族史"),
    # 排便习惯改变
    ("最近排便习惯改变了", "排便习惯改变"),
    ("大便变细了", "排便习惯改变"),
    ("便秘和腹泻交替出现", "排便习惯改变"),
]


@pytest.mark.parametrize("message,label", MUST_HIT)
def test_red_flag_fires(message, label):
    flags, instruction = safety.screen_input(message)
    assert label in flags, f"漏检报警症状：{message!r} 应命中「{label}」，实得 {flags}"
    assert instruction, "命中报警症状后必须返回注入指令"
    assert label in instruction


# --- Gate 1: ordinary questions must NOT fire ---------------------------------
# False positives cost one wasted referral line, so this list is short and only
# holds phrasings that would be actively annoying — including near-misses that
# share characters with a red flag (黑咖啡/黑木耳 vs 黑便, 献血/血糖 vs 便血).

MUST_NOT_HIT = [
    "益生菌要吃多久才有效",
    "每天该吃多少膳食纤维",
    "酸奶算不算益生菌",
    "宝宝可以吃益生菌吗",
    "吃了益生菌还是胀气怎么办",
    "益生菌需要冷藏吗",
    "菌株号是什么意思",
    "肠道菌群检测靠谱吗",
    "抗生素吃完要补益生菌吗",
    "益生元和益生菌区别",
    "乳糖不耐受怎么办",
    "什么是低FODMAP饮食",
    "压力大会影响肠道吗",
    "我最近老是放屁",
    "我大便不成形",
    "我最近有点便秘",
    "肚子胀气",
    "喝黑咖啡会拉肚子吗",
    "吃黑木耳有好处吗",
    "黑加仑对肠道好吗",
    "血糖高的人能吃益生菌吗",
    "献血后能吃益生菌吗",
    "我在减肥，益生菌有用吗",
    "肠癌筛查多大年龄开始做",
    "肠镜检查前要准备什么",
    "喝酸奶会胖吗",
]


@pytest.mark.parametrize("message", MUST_NOT_HIT)
def test_ordinary_question_does_not_fire(message):
    flags, instruction = safety.screen_input(message)
    assert flags == [], f"误报：{message!r} 不该触发报警症状，实得 {flags}"
    assert instruction is None


# --- Gate 1: known misses -----------------------------------------------------
# Each of these is a red flag stated the way Chinese actually orders the words.
# They are the phrasings most likely to arrive in production, and they miss.

FAMILY_HISTORY_MISSES = [
    "我爸得过肠癌，我要注意什么",
    "我妈是结直肠癌",
    "家里人得过肠癌",
    "我父亲直肠癌去世的",
    "爷爷有胃癌",
]


@pytest.mark.parametrize("message", FAMILY_HISTORY_MISSES)
@known_gap("RED_FLAGS 家族史 pattern is disease-first (肠癌…家族史) only, "
           "but Chinese states the relative first (我爸得过肠癌) — needs the "
           "mirrored alternative (亲属词).{0,14}(癌|息肉|克罗恩|溃疡性结肠炎)")
def test_family_history_relative_first(message):
    flags, _ = safety.screen_input(message)
    assert "消化道疾病家族史" in flags


WEIGHT_LOSS_MISSES = [
    "两个月掉了8斤，也没节食",
    "一个月掉了10斤",
    "没减肥掉了5公斤",
    "这半年轻了15斤",
    "饭量没变但一直在掉体重",
]


@pytest.mark.parametrize("message", WEIGHT_LOSS_MISSES)
@known_gap("RED_FLAGS 体重下降 anchors 掉了/轻了 to (体重|人|我) appearing FIRST, "
           "so the common 时间段+掉了N斤 form never matches — needs "
           "(掉|轻|少)了\\s*\\d+\\s*(斤|公斤|kg) on its own")
def test_weight_loss_without_subject_anchor(message):
    flags, _ = safety.screen_input(message)
    assert "体重下降" in flags


@pytest.mark.parametrize("message", ["夜里拉肚子", "夜里总是腹泻"])
@known_gap("RED_FLAGS 夜间症状 alternation lists 夜间/半夜/晚上/睡着 but not 夜里, "
           "which is the colloquial form; prompt.py SAFETY_RULES lists 夜间腹泻 "
           "as a referral trigger")
def test_night_symptom_colloquial_form(message):
    flags, _ = safety.screen_input(message)
    assert "夜间症状" in flags


# --- Gate 1: structure --------------------------------------------------------

def test_red_flag_table_is_well_formed():
    assert len(safety.RED_FLAGS) >= 10
    for entry in safety.RED_FLAGS:
        pattern, label = entry
        re.compile(pattern)
        assert label.strip(), "每条报警症状都要有中文标签，注入指令会把它念给模型听"
        assert not re.search(r"[A-Za-z]", label)


def test_multiple_flags_are_all_reported():
    flags, instruction = safety.screen_input("我最近大便带血，还瘦了十斤，晚上肚子疼醒")
    assert {"便血", "体重下降", "夜间症状"} <= set(flags)
    for flag in flags:
        assert flag in instruction


def test_mentions_self_separates_report_from_question():
    assert safety.mentions_self("我最近大便带血")
    assert safety.mentions_self("这几天一直拉肚子")
    assert not safety.mentions_self("便血一般是什么原因")
    assert not safety.mentions_self("黑便是怎么回事")


# --- The escalate-only merge (design constraint 1) ----------------------------
# The classifier is an LLM. It must never be able to talk the system down from a
# referral. These tests inject a classifier verdict directly, so they cover the
# merge logic without a network call.

BENIGN_ROUTING = {
    "risk": "normal", "category": "其他", "populations": ["通用"],
    "regulatory": False, "needs_current_info": False, "sub_queries": ["益生菌"],
}


def test_model_cannot_downgrade_a_self_reported_red_flag(monkeypatch):
    from agent import classify as classify_mod
    monkeypatch.setattr(classify_mod, "call_json", lambda *a, **k: dict(BENIGN_ROUTING))

    routing = classify_mod.classify("我最近大便带血，是不是吃益生菌就行了")
    assert routing["risk"] == "emergency", "模型判 normal 也不能压过确定性筛查"
    assert "便血" in routing["red_flags"]
    assert routing["safety_instruction"]


def test_emergency_survives_total_api_failure(monkeypatch):
    from agent import classify as classify_mod
    monkeypatch.setattr(classify_mod, "call_json", lambda *a, **k: None)

    routing = classify_mod.classify("我这两天一直吐，还吐出血了")
    assert routing["risk"] == "emergency"
    assert routing["_fallback"] is True
    assert "呕血" in routing["red_flags"]


def test_model_may_escalate_beyond_the_deterministic_floor(monkeypatch):
    from agent import classify as classify_mod
    escalated = dict(BENIGN_ROUTING, risk="emergency")
    monkeypatch.setattr(classify_mod, "call_json", lambda *a, **k: escalated)

    routing = classify_mod.classify("我肚子难受得厉害，说不清哪里不对")
    assert routing["risk"] == "emergency", "分类器只能升级，升级必须生效"


def test_third_person_red_flag_is_caution_not_emergency(monkeypatch):
    """Asking what 便血 means is not the same as reporting it."""
    from agent import classify as classify_mod
    monkeypatch.setattr(classify_mod, "call_json", lambda *a, **k: None)

    routing = classify_mod.classify("便血一般是什么原因引起的")
    assert routing["risk"] == "caution"
    assert routing["safety_instruction"], "科普问题也要带上结尾就医提示的指令"


# --- Gate 2: 违规宣称 ----------------------------------------------------------

BANNED = [
    ("这个菌株可以治疗肠易激综合征。", "宣称疾病治疗/治愈功能"),
    ("该菌株根治幽门螺杆菌。", "宣称疾病治疗/治愈功能"),
    ("长期服用可以预防肠癌。", "宣称疾病预防功能"),
    ("益生菌能替代药物治疗。", "宣称可替代药物或就医"),
    ("完全没有副作用，可以放心长期吃。", "宣称无副作用"),
    ("这个方法100%有效。", "绝对化用语"),
    ("保证有效，吃一个疗程就好。", "绝对化用语"),
    ("这是便秘的特效方法。", "夸大功效用语"),
    ("这是最有效的益生菌。", "绝对化最高级用语"),
]


@pytest.mark.parametrize("text,reason", BANNED)
def test_banned_claim_is_caught(text, reason):
    violations = safety.screen_output(text)
    assert violations, f"违规宣称漏检：{text!r}"
    assert reason in {r for _, r, _ in violations}


NEGATED = [
    "益生菌不能治疗肠炎，它只是辅助手段。",
    "任何食品都不得宣称预防疾病，这是《广告法》第十七条的规定。",
    "益生菌不能替代药物治疗，请遵医嘱。",
    "不要相信「100%有效」这类说法。",
    "商家宣称特效是违法的。",
    "并非没有副作用，部分人会腹胀。",
    "不能说这是最好的益生菌。",
]


@pytest.mark.parametrize("text", NEGATED)
def test_negated_claim_is_not_a_violation(text):
    """Explaining that a claim is illegal is the correct behaviour, not a breach."""
    assert safety.screen_output(text) == [], f"否定语境误报：{text!r}"


COMPLIANT = [
    "有一项人体试验显示该菌株有助于改善腹胀[1]。",
    "研究显示膳食纤维摄入与肠道菌群多样性相关[2]。",
    "这个说法目前只在动物实验里看到，人身上还没验证[3]。",
    "每日100亿CFU连续服用两周，成人受试者[1]。",
]


@pytest.mark.parametrize("text", COMPLIANT)
def test_compliant_phrasing_passes(text):
    assert safety.screen_output(text) == []


@pytest.mark.parametrize("text", [
    "没有任何产品可以根治肠易激。",
    "益生菌不替代药物，也不替代就医。",
])
@known_gap("_NEGATED misses bare 不X / 没有X, so a correct compliance statement "
           "trips the output gate and burns a repair round; after 2 rounds the "
           "user gets the canned failure text instead of a right answer")
def test_bare_negation_should_not_be_a_violation(text):
    assert safety.screen_output(text) == []


def test_violation_is_isolated_to_its_own_sentence():
    text = "益生菌不能治疗肠炎。但这款产品可以治愈便秘。"
    violations = safety.screen_output(text)
    assert len(violations) == 1
    matched, reason, sentence = violations[0]
    assert matched == "治愈"
    assert sentence == "但这款产品可以治愈便秘。"


def test_correction_names_the_offending_text():
    violations = safety.screen_output("这个菌株可以治疗肠易激，而且100%有效。")
    correction = safety.correction_for(violations)
    assert "治疗" in correction and "100%" in correction
    assert "有助于" in correction, "纠正指令要给出可用的替代说法，不能只说不行"


def test_banned_claim_table_is_well_formed():
    for pattern, reason in safety.BANNED_CLAIMS:
        re.compile(pattern)
        assert reason.strip()


# --- Gate 3: 急症固定话术 ------------------------------------------------------

def test_escalation_message_names_the_flags():
    message = safety.escalation_message(["便血", "体重下降"])
    assert "便血" in message and "体重下降" in message


def test_escalation_message_routes_to_care():
    message = safety.escalation_message(["剧烈腹痛"])
    assert "消化内科" in message
    assert "急诊" in message
    assert "不要自己开始或停止任何药物" in message


def test_escalation_message_refuses_to_diagnose():
    message = safety.escalation_message(["黑便或柏油样便"])
    assert "我没法替代这一步" in message
    assert "可能是" not in message, "急症话术不能猜病因"


def test_escalation_message_handles_empty_flags():
    message = safety.escalation_message([])
    assert "报警症状" in message
    assert len(message) > 100


def test_escalation_message_passes_its_own_compliance_gate():
    """The one answer no model writes still has to clear the rules the model does."""
    for flags in ([], ["便血"], ["便血", "呕血", "剧烈腹痛"]):
        assert safety.screen_output(safety.escalation_message(flags)) == []


@pytest.mark.parametrize("text_fn", [
    lambda: safety.escalation_message(["便血"]),
    lambda: safety.REFERRAL_INSTRUCTION.format(flags="便血"),
    lambda: safety.CORRECTION_INSTRUCTION.format(detail="「治疗」"),
])
def test_canned_text_is_chinese_and_not_ai_flavoured(text_fn):
    text = text_fn()
    assert re.search(r"[一-鿿]", text)
    # No emoji anywhere in shipped text.
    assert not re.search(r"[\U0001F300-\U0001FAFF☀-➿]", text)
    for filler in ["希望对您有帮助", "希望对你有帮助", "总的来说", "综上所述", "作为一个AI"]:
        assert filler not in text

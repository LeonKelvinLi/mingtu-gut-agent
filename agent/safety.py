"""Deterministic safety and compliance gates.

Two gates, both in code rather than prompt text, because a prompt instruction is
a suggestion and a regex is a guarantee:

  1. INPUT  — 消化道报警症状 screen. A hit does not hard-block the answer; it
     injects a mandatory instruction so the model leads with a referral. The
     model still judges whether the user is reporting a symptom or asking about
     one, which a hard block cannot do.

  2. OUTPUT — 中国广告法/食品安全法 claim screen. A hit is a FAILURE, not a
     warning: the caller regenerates with a correction. Widening this list to
     make an answer pass is the wrong fix.
"""

import re

# --- Gate 1: 报警症状 (alarm features warranting referral) --------------------
# Sources reconciled against research/04-limits.md. Each entry is (pattern, 说明).
# Patterns must match how people actually type, not how a textbook writes it.
# A user says "大便有血" and "瘦了好几斤", never "便血" and "不明原因体重下降".
# Every entry below is deliberately over-inclusive: a false positive costs one
# unnecessary "go see a doctor" line, a false negative can cost a missed cancer.
RED_FLAGS = [
    (r"便血|血便|"
     r"(大便|便便|大号|排便|粪便).{0,5}(有|带|出|见|混|夹).{0,3}血|"
     r"拉.{0,3}血|便后.{0,5}(出|滴|流).{0,3}血|"
     r"(厕纸|手纸|纸巾|纸上).{0,6}(有|带).{0,3}血|"
     r"肛门.{0,5}(出血|流血)|滴血|血丝",
     "便血"),
    (r"黑便|柏油|"
     r"(大便|便便|排便|粪便).{0,5}(发)?黑|"
     r"拉.{0,4}黑.{0,3}(便|屎|的)",
     "黑便或柏油样便"),
    (r"呕血|吐血|吐.{0,5}(出)?血|呕吐物.{0,5}血",
     "呕血"),
    # 「两个月掉了8斤」是最常见的说法，主语在时间状语后面甚至根本不出现，
    # 所以数量+单位的形式必须能独立命中，不能锚定在「体重/人/我」之后。
    (r"(体重|人|我).{0,8}(下降|减轻|轻了|掉了)|"
     r"(掉|轻|少|瘦)了\s*\d+\s*(斤|公斤|kg|KG|千克)|"
     r"瘦了|变瘦|消瘦|掉秤|掉体重|"
     r"不明原因.{0,8}(消瘦|体重|瘦)|"
     r"(没|未|不曾).{0,6}(减肥|节食).{0,10}(瘦|掉)",
     "体重下降"),
    (r"吞咽困难|咽不下|噎|吞咽.{0,4}(疼|痛)|"
     r"(咽|吞).{0,5}东西.{0,5}(疼|痛|困难)|食物.{0,5}卡",
     "吞咽困难"),
    (r"(持续|反复|一直|老是|总是|不停).{0,5}(呕吐|吐)|吐个不停|吐了好几",
     "持续呕吐"),
    (r"贫血|血红蛋白低|血色素低|(缺铁).{0,6}(性)?贫?血?",
     "贫血"),
    (r"(腹部|肚子|腹腔).{0,6}(包块|肿块|硬块|疙瘩)|"
     r"摸到.{0,6}(包块|肿块|硬块|疙瘩)",
     "腹部包块"),
    (r"(夜间|夜里|半夜|晚上|睡着|凌晨).{0,8}(痛醒|疼醒|腹痛|肚子疼|肚子痛|腹泻|拉肚子)",
     "夜间症状"),
    (r"黄疸|(皮肤|眼白|眼睛|巩膜).{0,5}(发)?黄",
     "黄疸"),
    (r"(发烧|发热|体温高).{0,14}(腹痛|肚子疼|肚子痛)|"
     r"(腹痛|肚子疼|肚子痛).{0,14}(发烧|发热|体温高)",
     "发热伴腹痛"),
    (r"(剧烈|严重|厉害).{0,5}(腹痛|肚子疼|肚子痛)|"
     r"(肚子|腹部).{0,5}(剧痛|绞痛)|"
     r"疼得.{0,8}(受不了|直不起|打滚|站不起|走不了)",
     "剧烈腹痛"),
    (r"脱水|"
     r"(一天|一整天|很久|好久).{0,6}(没|未).{0,4}(尿|小便)|"
     r"尿.{0,4}(很少|特别少|没有)",
     "脱水"),
    # 中文习惯先说亲属再说病名（我爸得过肠癌），所以两个方向都要匹配。
    # 只写「病名…家族史」这一种顺序，最常见的口语形式反而全部漏掉。
    (r"(肠癌|结直肠癌|大肠癌|胃癌|食道癌|食管癌|消化道肿瘤|息肉|克罗恩|溃疡性结肠炎)"
     r".{0,14}(家族史|家里|父母|直系|我爸|我妈|爷爷|奶奶|遗传)|"
     r"(我爸|我妈|父亲|母亲|爷爷|奶奶|外公|外婆|哥|姐|弟|妹|家里人|家人|亲戚|直系)"
     r".{0,14}(肠癌|结直肠癌|大肠癌|胃癌|食道癌|食管癌|消化道肿瘤|息肉|克罗恩|溃疡性结肠炎)",
     "消化道疾病家族史"),
    (r"排便习惯.{0,8}(改变|变化|不一样)|"
     r"(大便|便便).{0,6}(变细|变形|形状.{0,4}(变|改))|"
     r"(便秘|腹泻|拉肚子).{0,8}交替",
     "排便习惯改变"),
]

# 症状归因于自身的表述。用于区分"我有便血"和"便血是什么原因"。
# 命中与否都会注入指令，此处仅用于给模型更明确的信号。
_FIRST_PERSON = re.compile(r"我|自己|本人|最近|这几天|这两天|一直|老是|总是")

REFERRAL_INSTRUCTION = (
    "【安全指令】用户消息中出现了消化道报警症状（{flags}）。"
    "如果用户是在描述自己的症状，你必须把就医建议放在回答最前面，明确说明这类症状需要"
    "消化科面诊并完善相应检查，不要先给饮食或益生菌建议，也不要推测具体疾病。"
    "如果用户只是在询问该症状的科普知识而非描述自身情况，正常科普即可，"
    "但仍需在结尾提示出现该症状本人应及时就医。"
)


def screen_input(text):
    """Return (hit_flags, instruction_or_None) for a user message."""
    hits = []
    for pattern, label in RED_FLAGS:
        if re.search(pattern, text):
            hits.append(label)
    if not hits:
        return [], None
    return hits, REFERRAL_INSTRUCTION.format(flags="、".join(hits))


# 纯知识性提问的框架。只有确证是在问知识、且没有求助意味时才降级。
_KNOWLEDGE_FRAME = re.compile(
    r"是什么原因|是什么意思|为什么会|怎么形成|"
    r"科普|网上说|听说|新闻说|文章说|书上说|"
    r"一般(是|由|来说)|通常(是|由)|医学上|临床上|"
    r"(是不是|算不算)[^。！？]{0,10}(吗|呢)$"
)
# 求助意味：出现这些就算带着问句框架也不能降级。
_HELP_SEEKING = re.compile(r"怎么办|该怎么|要不要|需不需要|急|严重吗|要紧吗|危险吗|吓|担心|害怕")
# 第三人称转述。仍然要升级（家人确实该就诊），但话术得换称谓。
_THIRD_PERSON = re.compile(
    r"我(妈|爸|母亲|父亲|老婆|老公|妻子|丈夫|儿子|女儿|孩子|奶奶|爷爷|外婆|外公|哥|姐|弟|妹|朋友|同事)|"
    r"妈妈|爸爸|母亲|父亲|老婆|老公|妻子|丈夫|儿子|女儿|孩子|宝宝|"
    r"奶奶|爷爷|外婆|外公|婆婆|公公|家(里)?人|亲戚|朋友|同事|他|她"
)


def mentions_self(text):
    """这条消息是不是在描述某个人身上真实发生的症状。

    中文是主语脱落语言：「便血三天了」「拉血了」「黑便两天」「吐血了怎么办」
    都不含任何人称词。原来靠「有没有出现我/最近/一直」来判断，结果是
    真人自述被降级成一般咨询，走完整作答管线，一句就医提示都没有——
    这是这套系统里后果最严重的一类漏判。

    所以判定方向反过来：命中报警症状时**默认按自述处理**，
    只有能确证是在问知识、且不带求助意味时才降级。
    宁可多给一次就医提示，不能漏掉一次真实求助。
    """
    if _THIRD_PERSON.search(text) or _FIRST_PERSON.search(text):
        return True
    if _KNOWLEDGE_FRAME.search(text) and not _HELP_SEEKING.search(text):
        return False
    return True


def is_third_person(text):
    """症状是别人的（家人、朋友），话术要换称谓。"""
    return bool(_THIRD_PERSON.search(text)) and not re.search(r"^我[^妈爸母父老儿女孩奶爷外哥姐弟妹朋同]", text)


# --- Gate 2: 违规宣称 (prohibited health claims) ------------------------------
# 《广告法》第十七条：非医疗广告不得涉及疾病治疗功能。
# 《食品安全法》第七十三条：食品广告不得涉及疾病预防、治疗功能。
# 主语词：食品/益生菌类。把疾病功能归给这些主语才是法规禁止的宣称。
_SUBJ = r"(?:益生菌|益生元|合生元|本品|该品|该产品|这款|这个|这些|那个|产品|菌株|菌种|酸奶|发酵乳|保健食品|膳食补充剂|它)"
# 疾病/症状词：不能只认「病癌症炎」四个字，本领域最高频的恰恰是腹泻便秘感染复发。
_DISEASE = (r"(?:病|癌|症|炎|感染|腹泻|便秘|胀气|口臭|湿疹|过敏|复发|溃疡|息肉|"
            r"肿瘤|幽门螺杆菌|肠易激|克罗恩|脂肪肝|糖尿病|肥胖|失眠|焦虑|抑郁)")

BANNED_CLAIMS = [
    # 治疗类：必须带主语。裸「治疗」在这个产品里是绕不开的正当表述
    # （「接受规范治疗」「治疗方案由医生决定」），一刀切会把就医导航本身拦死。
    (rf"{_SUBJ}[^。！？；，,、\n]{{0,12}}(治疗|治愈|根治|治好|断根|除根|痊愈|药到病除|主治)",
     "宣称疾病治疗/治愈功能"),
    (r"(治疗|治愈|根治|治好)[^。！？；\n]{0,6}(作用|功效|效果|能力)",
     "宣称疾病治疗/治愈功能"),
    # 口语化的疗效断言。书面语词表漏掉的正是用户和商家真正会说的话。
    (r"(疗效|有效率|治愈率|见效快|立竿见影|永不复发|不再复发|彻底好了|根除病根|断根|除根|治好了)",
     "宣称疗效"),
    # 预防类：正序 + 反序 + 防癌抗癌单列。宾语不限于「病癌症炎」四个字。
    (rf"(预防|防治|抵御|防止)(?!接种)[^。！？；，,、\n]{{0,10}}{_DISEASE}",
     "宣称疾病预防功能"),
    (rf"{_DISEASE}[^。！？；，,、\n]{{0,8}}(预防|防护)(作用|效果|功能)",
     "宣称疾病预防功能"),
    (r"(防癌|抗癌|抑癌|抗肿瘤|防肿瘤|抑制肿瘤)", "宣称抗癌功能"),
    (r"(替代|代替|不用再吃|不必吃)[^。！？；\n]{0,6}(药物|药品|药|治疗|就医|看医生)",
     "宣称可替代药物或就医"),
    (r"(无|没有|零|不含)[^。！？；\n]{0,4}(副作用|毒副作用|依赖性)", "宣称无副作用"),
    (r"100%|百分之百|绝对(有效|安全|放心)|保证(有效|治好|见效)|包治|人人有效",
     "绝对化用语"),
    (r"(特效|神效|秘方|奇效|神药|圣品)", "夸大功效用语"),
    (r"最(好|佳|强|有效|顶级)的?(益生菌|产品|菌株|品牌)", "绝对化最高级用语"),
]

# 允许的语境：引用法规原文、说明什么是违法宣称、或明确否定该宣称时不算违规。
# 「治疗」在合法语境里出现得极多：用户正在接受的治疗、医生的治疗方案、
# 药物治疗、根除治疗期间。法规禁止的是**把疾病治疗功能归给食品**，
# 而不是禁止提到治疗这两个字。不做这个区分，闸门会把
# 「治疗前提前吃」这种完全正当的句子拦下来，白白消耗重写轮次。
_CLINICAL_CONTEXT = re.compile(
    r"(治疗|用药)(期间|前|后|过程|方案|结束|开始|同时)|"
    r"(根除|规范|药物|抗生素|联合|三联|四联|临床|医生|医院|正规|常规|辅助)治疗|"
    r"(接受|配合|完成|进行|开始|结束|停止)治疗|"
    r"治疗(的)?(同时|基础上|之外)|"
    r"不能替代(药物|治疗|就医)|非治疗"
)

_NEGATED = re.compile(
    r"不能|不可|不得|不会|无法|禁止|违法|不合规|并非|不是|"
    r"不应|切勿|不要|谈不上|没有证据|缺乏证据|不能宣称|属于违规|"
    # 裸否定。少了这些，「没有任何产品可以根治肠易激」这种完全正确的合规表述
    # 会被自己的合规闸门拦下，白白消耗重写轮次，两轮之后用户拿到的是兜底话术
    # 而不是正确答案——闸门太紧和太松一样是缺陷。
    r"不[替代治预根]|没有[任何一]|没有.{0,4}(产品|办法|方法)|"
    r"不等同|不等于|不代表|不意味|不构成|不属于|不能算|称不上|而非|并不"
)


def screen_output(text):
    """Return a list of (matched_text, reason) for prohibited claims.

    A match inside a sentence that negates or forbids the claim is not a
    violation — the agent explaining that "益生菌不能治疗肠炎" is correct
    behaviour, not a breach.
    """
    violations = []
    for sentence in re.split(r"(?<=[。！？；;.!?\n])", text):
        if not sentence.strip():
            continue
        for pattern, reason in BANNED_CLAIMS:
            for m in re.finditer(pattern, sentence):
                # 临床豁免同样要限定在违规词所在的分句：整句豁免会让
                # 「不能替代药物，本品可以根治肠癌」的后半句蹭到前半句的免罪符。
                if reason.startswith("宣称疾病治疗") and _clinical_for(sentence, m):
                    continue
                # 否定必须作用在这个违规词上，而不是「同一句里出现过否定词」。
                # 原来的整句作用域等于给了一个万能开关：只要句子里有任意否定词，
                # 同句所有违规宣称一起被放行——一句话塞五条违规也能全身而退。
                if _negated_for(sentence, m):
                    continue
                term = next((g for g in reversed(m.groups() or ()) if g), m.group(0))
                violations.append((term, reason, sentence.strip()))
                break          # 同一条规则在一句里只报一次
    return violations


# 否定词的作用半径：只看违规词前面这么多字，且不跨越分句边界。
_NEGATION_WINDOW = 14
# 元语境：句子在讲「这种说法违法/不合规」，而不是在做这种说法。
_META_CONTEXT = re.compile(
    r"违法|违规|不合规|不得|禁止|不允许|属于虚假|涉嫌|监管|法规|广告法|食品安全法|"
    r"这类(说法|宣称|表述)|不能宣称|不应宣称")
_CLAUSE_BREAK = re.compile(r"[，,；;。！？\n、]")


def _negated_for(sentence, m):
    """判断这个违规词是否真的落在某个否定的作用域内。

    两处都要看：
      1 匹配片段内部——「益生菌不能治疗肠炎」的否定夹在主语和动词之间，
        只看左边会漏掉，结果把一句正确的合规说明当成违规。
      2 匹配片段左侧的一个短窗口，且不跨越逗号顿号分号——
        「益生菌不能替代药物，本品可以根治肠癌」里的「不能」不该给后半句免罪。
    """
    if _NEGATED.search(m.group(0)):
        return True
    # 讨论「这么说是违法的」属于合规科普，不是违规宣称本身。
    if _META_CONTEXT.search(sentence):
        return True
    start = m.start()
    # 紧邻前一个字的裸否定：「不替代药物」里的「不」贴着违规词，
    # 落在窗口之外的边界上，按窗口取会整个漏掉。
    if start and sentence[max(0, start - 2):start].strip().endswith(("不", "无", "非", "别", "勿")):
        return True
    left = sentence[:start]
    break_pos = None
    for m in _CLAUSE_BREAK.finditer(left):
        break_pos = m.end()
    clause_start = break_pos if break_pos is not None else 0
    window_start = max(clause_start, start - _NEGATION_WINDOW)
    return bool(_NEGATED.search(left[window_start:]))


def _clinical_for(sentence, m):
    """违规词所在分句里是否是临床语境（医生的治疗，而非食品的功能宣称）。"""
    left, right = sentence[:m.start()], sentence[m.end():]
    lb = [x.end() for x in _CLAUSE_BREAK.finditer(left)]
    rb = _CLAUSE_BREAK.search(right)
    clause = sentence[(lb[-1] if lb else 0): m.end() + (rb.start() if rb else len(right))]
    return bool(_CLINICAL_CONTEXT.search(clause))


CORRECTION_INSTRUCTION = (
    "【合规指令】你上一条回答中出现了中国法规不允许的健康宣称：{detail}。"
    "请重写这条回答：保留全部科学内容和证据等级标注，但删除或改写上述表述。"
    "食品与益生菌只能表述为'有助于''可能改善''研究显示与……相关'，"
    "不得涉及疾病的预防、治疗、治愈功能，不得使用绝对化用语。"
)


def correction_for(violations):
    detail = "；".join(f"「{m}」（{reason}）" for m, reason, _ in violations)
    return CORRECTION_INSTRUCTION.format(detail=detail)


# --- Gate 3: 急症固定应答 -----------------------------------------------------
# Deliberately not generated. When the classifier says emergency, this text ships
# verbatim before any model output. A retrieval miss, a refusal, an API timeout,
# a prompt-injection attempt — none of them can suppress it, because no model is
# in the path. This is the one answer in the system the model does not write.

ESCALATION_TEMPLATE = """你描述的情况里有需要尽快就医的信号（{flags}）。

这类症状不适合先用饮食或益生菌调理，需要消化科当面检查来判断原因。建议你：

1. 尽快挂消化内科门诊；如果出现大量便血、呕血、剧烈腹痛不缓解、意识模糊或站起来发晕，
   直接去急诊，不要等门诊号。
2. 就诊时把这几件事告诉医生：症状是什么时候开始的、多久发作一次、大便的颜色和形状、
   体重最近有没有变化、正在吃哪些药（包括保健品）。
3. 在医生看过之前，不要自己开始或停止任何药物。

我可以帮你做的是解释检查项目大概是做什么的、帮你想好要问医生什么问题。
但具体是什么原因、要怎么治，需要医生结合检查来判断，我没法替代这一步。"""


def escalation_message(flags):
    return ESCALATION_TEMPLATE.format(flags="、".join(flags) if flags else "报警症状")

"""当前 Skill 调用所属任务的命名空间（评测任务产生的缓存与用户数据隔离）。

执行器在每次技能调用前设置（取自 RunContext.namespace），技能内部据此给持久化缓存加 provenance，
RSI 等下游只消费用户命名空间（空字符串）的记录。评测命名空间的键形如 eval-<run>-<route>。
"""
from contextvars import ContextVar

current_namespace: ContextVar[str] = ContextVar("advisor_namespace", default="")

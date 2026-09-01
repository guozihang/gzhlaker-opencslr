# -*- coding: utf-8 -*-
"""模型构建函数注册表。

新增模型时,在 build_function 的构建函数上添加
``@register_model("name")`` 装饰器即可,config 中通过
``model: name`` 引用,无需修改任何 import 列表或工厂分支。
"""

MODEL_BUILDERS = {}


def register_model(name):
    """注册模型构建函数的装饰器。

    Args:
        name: 模型注册名(config 中 ``model:`` 字段使用的名字)。

    Returns:
        装饰器函数。
    """

    def decorator(fn):
        if name in MODEL_BUILDERS:
            raise ValueError(f"Model {name!r} is already registered.")
        MODEL_BUILDERS[name] = fn
        return fn

    return decorator


def get_model_builder(name):
    """按注册名获取模型构建函数。

    Args:
        name: 模型注册名。

    Returns:
        callable: 模型构建函数。

    Raises:
        ValueError: 注册名不存在时抛出,并列出全部可用模型。
    """
    if name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model {name!r}. Available: {sorted(MODEL_BUILDERS)}."
        )
    return MODEL_BUILDERS[name]

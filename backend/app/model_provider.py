"""模型服务提供者模块 - 统一管理 LLM、Embedding、Rerank 等模型的初始化与获取。

所有模型实例通过工厂函数懒加载，支持通过配置切换模型名称。
"""

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings


def get_chat_model(model_name: str | None = None, temperature: float = 0.2, settings: Settings | None = None) -> ChatOpenAI | None:
    """获取对话模型实例（阿里云 DashScope 兼容的 OpenAI 接口）。
    
    Args:
        model_name: 可选，指定模型名称，默认使用配置中的 chat_model。
        temperature: 温度参数，控制输出的随机性。
        settings: 可选，传入配置覆盖默认值。
        
    Returns:
        ChatOpenAI 实例，如果 API Key 未配置则返回 None。
    """
    value = settings or get_settings()
    if not value.dashscope_api_key or value.dashscope_api_key == "请用户自行填写":
        return None
    model = model_name or value.chat_model
    return ChatOpenAI(
            model=model,
            api_key=value.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=None if "kimi" in model else temperature,
            extra_body={"enable_thinking": False},
        )


def get_judge_model(model_name: str | None = None, temperature: float = 0.2, settings: Settings | None = None) -> ChatOpenAI | None:
    """获取评估裁判模型实例。
    
    裁判模型使用独立配置的 API Key 和 Base URL，与对话模型解耦。
    
    Args:
        model_name: 可选，指定模型名称。
        temperature: 温度参数。
        settings: 可选，传入配置覆盖默认值。
        
    Returns:
        ChatOpenAI 实例，如果裁判模型未配置则返回 None。
    """
    value = settings or get_settings()
    model=model_name or value.judge_model
    if not value.judge_api_key or not value.judge_base_url or not model:
        print("Judge model is not configured. Please set JUDGE_MODEL, JUDGE_API_KEY, and JUDGE_BASE_URL in the .env file.")
        return None

    return ChatOpenAI(
            model=model,
            api_key=value.judge_api_key,
            base_url=value.judge_base_url,
            temperature=temperature
        )


def get_embedding_model(settings: Settings | None = None) -> DashScopeEmbeddings | None:
    """获取文本向量化模型实例。
    
    Args:
        settings: 可选，传入配置覆盖默认值。
        
    Returns:
        DashScopeEmbeddings 实例，如果 API Key 未配置则返回 None。
    """
    value = settings or get_settings()
    if not value.dashscope_api_key or value.dashscope_api_key == "请用户自行填写":
        return None
    return DashScopeEmbeddings(model=value.embedding_model, dashscope_api_key=value.dashscope_api_key)


if __name__ == "__main__":
    # 调试用：测试裁判模型的结构化输出能力
    model = get_judge_model()
    from app.talent_evaluation_dispatch import EvaluationDimensionPlan, DIMENSION_GENERATOR_PROMPT

    res = model.with_structured_output(EvaluationDimensionPlan).invoke(
                [
                    ("system", DIMENSION_GENERATOR_PROMPT),
                    ("user", "你好"),
                ]
            )
    print(res)
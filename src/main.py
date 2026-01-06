import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 页面配置
st.set_page_config(
    page_title="Law_LoRA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 模型路径
model_path = "/data/linfengyun/models/Qwen/Qwen2.5-7B-Instruct"
lora_path = "./output/Qwen2.5-7B-Instruct_lora/checkpoint-1875"
device = "cuda:7"

# 模型加载
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 7},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()

    model = PeftModel.from_pretrained(
        base_model,
        model_id=lora_path
    ).to(device)

    return tokenizer, model


tokenizer, model = load_model()

# 推理函数
def generate_response(messages, max_new_tokens=512):
    # 加 system prompt
    chat_messages = [
        {"role": "system", "content": "假设你是一名专业法律从业者"}
    ] + messages

    prompt = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        [prompt],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
        )

    generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]

    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return response


# CSS 美化
st.markdown(
    """
    <style>
    .title {font-size:42px;font-weight:700;text-align:center;margin-bottom:30px;}
    .chat-box {max-width:900px;margin:0 auto;}
    .user-msg {background:#ffecec;padding:14px 18px;border-radius:12px;margin:10px 0;text-align:right;}
    .bot-msg {background:#f5f7fa;padding:14px 18px;border-radius:12px;margin:10px 0;}
    .stTextInput > div > div > input {border-radius:20px;padding:12px 16px;}
    .sidebar-title {font-size:22px;font-weight:600;margin-bottom:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# 侧边栏
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Model Configuration</div>", unsafe_allow_html=True)

    st.text_input("Base Model", model_path)
    st.text_input("LoRA Path", lora_path)
    st.text_input("Max Tokens", str(512))

# 会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 主页面
st.markdown("<div class='title'>法律知识问答</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 输入
user_input = st.chat_input("请输入法律问题，如：离婚时夫妻共同财产如何分割？")

if user_input:
    # 用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # 模型推理
    with st.spinner("模型正在思考..."):
        response = generate_response(
            st.session_state.messages,
            max_new_tokens=512
        )

    # 模型回复
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()

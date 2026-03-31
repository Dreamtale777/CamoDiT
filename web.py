import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
from omegaconf import OmegaConf
import torchvision.transforms as transforms

# 引用项目自带的函数
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="伪装目标检测系统",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS样式 ====================
st.markdown("""
<style>
    /* 全局样式 */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f4037, #99f2c8);
        border-radius: 12px;
        color: white;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    /* 卡片容器 */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.1);
    }
    /* 图片容器 */
    .image-wrapper {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        background-color: #fafafa;
        text-align: center;
        padding: 0.5rem;
    }
    .image-wrapper img {
        border-radius: 8px;
        max-width: 100%;
        height: auto;
    }
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #2b5876, #4e4376);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(135deg, #1f3b4f, #3a2c5a);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: white;
    }
    /* 上传区域美化 */
    .stFileUploader > div:first-child {
        border: 2px dashed #cccccc;
        border-radius: 20px;
        background-color: #f9f9fb;
        padding: 2rem;
        transition: all 0.2s ease;
    }
    .stFileUploader > div:first-child:hover {
        border-color: #4e4376;
        background-color: #f0f0f5;
    }
    /* 侧边栏 */
    .sidebar-info {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }
    .sidebar-info p {
        margin: 0;
        font-size: 0.9rem;
    }
    hr {
        margin: 1rem 0;
    }
    /* 页脚 */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #6c757d;
        font-size: 0.8rem;
        border-top: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 顶部标题 ====================
st.markdown("""
<div class="main-header">
    <h1>🕵️ 伪装目标检测系统</h1>
    <p>基于扩散模型的智能伪装识别 | 上传图片，一键寻找隐藏目标</p>
</div>
""", unsafe_allow_html=True)

# ==================== 侧边栏信息 ====================
with st.sidebar:
    st.markdown("### ℹ️ 系统信息")
    device_info = "🚀 GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
    st.markdown(f"**运行设备**: {device_info}")
    st.markdown("**模型**: CamoDiT")
    st.markdown("**算法**: 条件扩散模型")
    st.markdown("---")
    st.markdown("### 📌 使用说明")
    st.markdown("1. 点击上传按钮选择图片\n2. 点击「开始寻找目标」\n3. 等待几秒，右侧将显示检测结果")
    st.markdown("---")
    st.markdown("### 🔍 提示")
    st.markdown("本系统针对伪装目标（如迷彩动物、隐藏物体）进行检测，输出为高亮区域的二值掩码。")
    st.markdown("---")
    st.markdown("### 📧 联系")
    st.markdown("如有问题，请联系开发团队。")

# ==================== 模型加载（缓存） ====================
@st.cache_resource
def load_model():
    config_path = 'config/camoDiffusion_352x352.yaml'
    model_path = 'model-best/model-best.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return None, None, "❌ 没找到配置文件或模型文件，请检查路径。"

    cfg = OmegaConf.load(config_path)
    img_size = cfg.diffusion_model.params.image_size

    try:
        cond_uvit = instantiate_from_config(
            cfg.cond_uvit,
            conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass)
        )
        model_arch = recurse_instantiate_from_config(cfg.model, unet=cond_uvit)
        diffusion_model = instantiate_from_config(cfg.diffusion_model, model=model_arch).to(device)

        state_dict = torch.load(model_path, map_location=device)
        if 'model' in state_dict:
            diffusion_model.load_state_dict(state_dict['model'])
        else:
            diffusion_model.load_state_dict(state_dict)

        diffusion_model.eval()
        return diffusion_model, img_size, "✅ 模型加载成功，系统已就绪！"
    except Exception as e:
        return None, None, f"⚠️ 模型加载失败: {str(e)}"

model, img_size, status_msg = load_model()

# 显示模型状态（侧边栏）
with st.sidebar:
    if "成功" in status_msg:
        st.success(status_msg)
    elif "失败" in status_msg:
        st.error(status_msg)
    else:
        st.info(status_msg)

if model is None:
    st.error(status_msg)
    st.stop()

# ==================== 主内容区域 ====================
uploaded_file = st.file_uploader(
    "📤 点击或拖拽图片至此区域",
    type=["jpg", "png", "jpeg"],
    help="支持 JPG、PNG、JPEG 格式"
)

if uploaded_file is not None:
    # 读取原图
    orig_img = Image.open(uploaded_file).convert('RGB')

    # 左右两列布局（比例稍作调整）
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📸 原始图片")
        st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
        st.image(orig_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🎯 检测结果")
        # 按钮放在卡片内，样式统一
        if st.button("🔍 开始寻找目标", type="primary", use_container_width=True):
            with st.spinner('🧠 正在逐层剥离伪装，请稍候...'):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # 图片预处理
                transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor()
                ])
                img_data = transform(orig_img).unsqueeze(0).to(device)
                img_data = (img_data - 0.5) * 2   # 归一化到[-1,1]

                # 模型推理
                with torch.no_grad():
                    out_images = model.sample(img_data)
                    final_pic = out_images[-1].clamp(0, 1).squeeze().cpu().numpy()

                # 转换为显示图像
                final_pic_disp = (final_pic * 255).astype(np.uint8)
                res_img = Image.fromarray(final_pic_disp, mode='L')

                # 显示结果
                st.markdown('<div class="image-wrapper">', unsafe_allow_html=True)
                st.image(res_img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # 添加一个成功提示
                st.success("✨ 检测完成！右侧为伪装目标的二值掩码。")
        else:
            # 未点击按钮时显示占位符
            st.markdown('<div class="image-wrapper" style="background:#f0f2f6; min-height:200px; display:flex; align-items:center; justify-content:center;">', unsafe_allow_html=True)
            st.markdown('<p style="color:#6c757d; text-align:center;">⬅️ 点击按钮开始检测</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # 未上传图片时的友好提示
    st.markdown('<div class="card" style="text-align:center; padding:2rem;">', unsafe_allow_html=True)
    st.markdown("### 🌟 等待图片上传")
    st.markdown("请从左侧上传一张图片，系统将自动识别其中的伪装目标。")
    st.markdown("---")
    st.markdown("**支持格式**: JPG, PNG, JPEG | **推荐分辨率**: 352x352 或相近比例")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== 页脚 ====================
st.markdown("""
<div class="footer">
    <p>伪装目标检测系统 | 基于扩散模型 | 仅供研究演示使用</p>
</div>
""", unsafe_allow_html=True)
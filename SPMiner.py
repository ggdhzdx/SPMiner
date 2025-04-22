import os
import re
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from decimer_segmentation import (
    segment_chemical_structures,
    segment_chemical_structures,
)
import pathlib
from PyPDF2 import PdfWriter, PdfReader
import dashscope
from DECIMER import predict_SMILES
from rdkit import Chem
from rdkit.Chem import Draw
import requests  # 用于发送HTTP请求
import json  # 用于解析和生成JSON格式数据
import pdfplumber  # 用于读取PDF文件
import fitz  # PyMuPDF
import hashlib
import zipfile
import io
from collections import defaultdict
import csv
import time

allapi = {
    "mineru": "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1MzMwMzgzOCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc0MzA2NjgwOCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTk4NTU4MzI5MzIiLCJvcGVuSWQiOm51bGwsInV1aWQiOiIzZjRiZTU1Yy1hZTM4LTQ5MDQtODUxNy05MDY5ODA4M2ZhMGMiLCJlbWFpbCI6IiIsImV4cCI6MTc0NDI3NjQwOH0.fuP98caz394-3LXyAfGSqHc-vgNZTDBIUo91UxDlc8M4gtCwmCrBgFCmw_UAD_5xxlyYABaTXpbtzFTFKaTxLA",
    "qwen":"sk-4f7eae96162a4db99e74842c54ee7888",
    "deepseek":"sk-6946dfacf45f48f3a52427903404ff82"
    # 其他 API 配置可以在这里添加
}
file_note={
    "file_name":"test.pdf",
    "file_path":"/home/stu/yc/decimer/test.pdf"
}
os.path.splittext(fi)[-1]
MODELS = [
    {"name": "qwen-max", "type": "dashscope"},
    {"name": "deepseek-chat", "type": "deepseek"}
]  # 对比模型配置
params = ['化合物名称','吸收波长(λabs)','发射波长(λem)','半峰宽(FWHM)','s1态能级(Es1)','t1态能级(Et1)','带隙(Egap)','HOMO能级','LUMO能级','PLQY(光致发光量子产量)','delta_Est(单三线态能差)',"CIE(色坐标)","EQE(外量子效率)","器件结构","knr(非辐射速率)"]
image_folder = "/home/stu/yc/decimer/pdf_images"#提取图片保存到该位置
raw_name="BN-TP"
poppler_path = "/home/software/anaconda3/envs/decimer/bin"


def pdf2md(file_note: dict, allapi: dict) -> string:
    """输入pdf，得到md文件"""
    mineru_api = allapi.get("mineru")
    file_name = file_note.get("file_name")
    url = 'https://mineru.net/api/v4/file-urls/batch'
    header = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {mineru_api}"
    }
    data = {
        "enable_formula": True,
        "language": "en",
        "layout_model": "doclayout_yolo",
        "enable_table": True,
        "files": [
            {"name": f"{file_name}", "is_ocr": True, "data_id": "abcd"}
        ]
    }
    # 第一步：获取文件上传链接
    response = requests.post(url, headers=header, data=json.dumps(data))
    result = response.json()

    file_path = file_note["file_path"]
    file_urls = result["data"]["file_urls"]
    
    # 第二步：上传 PDF 文件
    with open(file_path, 'rb') as f:
        res_upload = requests.put(file_urls[0], data=f)
        if res_upload.status_code != 200:
            raise Exception(f"文件上传失败: {res_upload.status_code}")

    batch_id = result["data"]["batch_id"]
    url = f'https://mineru.net/api/v4/extract-results/batch/{batch_id}'

    # 轮询检查处理结果
    start_time = time.time()
    timeout = 60  # 超时时间60秒
    while True:
        # 轮询查询 API
        response = requests.get(url, headers=header)
        back_information = response.json()

        # 获取 full_zip_url 的正确方式
        extract_result = back_information.get('data', {}).get('extract_result', [])
        if extract_result:
            full_zip_url = extract_result[0].get('full_zip_url')
            
            if full_zip_url:
                zip_response = requests.get(full_zip_url)
                
                # 解压 ZIP 文件
                with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zip_ref:
                    zip_ref.extractall("extracted_files")  # 解压到本地目录
                
                # 查找并读取 Markdown 文件
                md_content = ""
                for file_name in os.listdir("extracted_files"):
                    if file_name.endswith(".md"):
                        with open(os.path.join("extracted_files", file_name), "r", encoding="utf-8") as md_file:
                            md_content = md_file.read()
                        break  # 读取第一个 Markdown 文件后退出

                if md_content:
                    print("Markdown 文件已正确储存在变量 md_content 中")
                    return md_content
                else:
                    print("未找到 Markdown 文件")
                    return None

        # 检查超时
        if time.time() - start_time > timeout:
            print("超时，未获得 full_zip_url")
            return back_information  # 返回最后一次的结果

        # 等待3秒后继续检测
        time.sleep(3)

# 调用函数
md_content = pdf2md(file_note, allapi)

class MD2CSV:
    def __init__(self, md_content, MODELS, params, allapi):
        self.md_content = md_content
        self.MODELS = MODELS
        self.params = params
        self.allapi = allapi
        self.headers = []
        self.all_models_data = []
        self.global_conditions = defaultdict(set)
        
        # 初始化处理管道
        self.promote_gen = {
            "qwen": self.__gen_promote4qwen,
            "deepseek": self.__gen_promote4deepseek
        }
        self.model_run = {
            "qwen": self.__run_qwen,
            "deepseek": self.__run_deepseek
        }
        self.string2csv = {
            "qwen": self.__parse_qwen,
            "gpt": self.__parse_deepseek
        }
        
    def __gen_promote4qwen(self):
        return (
                 f"请从以下文本中提取化合物的名称、性质、测试条件、测试结果及其单位，按照下面的格式输出（不要添加其他额外的标记）。如果某个参数、单位或测试条件未能检索到，请标记为NA。如果有其他额外的参数，附在后边，且保持与前面参数一致的格式。\n" +
                 f"输出格式：\n" +
                 f"测试条件: 测试结果: 单位: \n" +
                 f"请按照上述格式逐一输出每个检索到的参数。例如: \n" +
                 f"发射波长(λem): (1×10^(-5)M, 298K, toluene): 523: nm \n" +
                 f"发射波长(λem): PhCzBCz doped film (3wt% doping concentration): 528: nm\n" +
                 f"参数包括:{', '.join(params)}\n文本：{md_content}"
        )

    def __gen_promote4deepseek(self):
        return (
                 f"请从以下文本中提取化合物的名称、性质、测试条件、测试结果及其单位，按照下面的格式输出（不要添加其他额外的标记）。如果某个参数、单位或测试条件未能检索到，请标记为NA。如果有其他额外的参数，附在后边，且保持与前面参数一致的格式。\n" +
                 f"输出格式：\n" +
                 f"测试条件: 测试结果: 单位: \n" +
                 f"请按照上述格式逐一输出每个检索到的参数。例如: \n" +
                 f"发射波长(λem): (1×10^(-5)M, 298K, toluene): 523: nm \n" +
                 f"发射波长(λem): PhCzBCz doped film (3wt% doping concentration): 528: nm\n" +
                 f"参数包括:{', '.join(params)}\n文本：{md_content}"
        )
   
    def __run_qwen(self, model_name):
        try:
            response = dashscope.Generation.call(
                api_key=self.allapi.get("qwen"),
                model=model_name,
                messages=[{'role': 'user', 'content': self.promote_gen["qwen"]()}],
                result_format='text'
            )
            return response.output.text.strip() if response and response.output else ""
        except Exception as e:
            print(f"Qwen接口错误: {str(e)}")
            return ""

    def __run_deepseek(self, model_name):
        try:
            # 定义 API 请求参数
            deepseek_api=allapi.get("deepseek")
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {deepseek_api}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "deepseek-chat",
                "messages":[
                    {"role": "system", "content": "你是一个专业的化学信息提取助手，请按照下面的格式回答问题"},
                    {"role": "user", "content": self.promote_gen["deepseek"]()}
                ],
            "stream": False
            }
            # 发送请求
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # 自动处理 HTTP 错误码
    
            # 解析响应
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"deepseek接口错误: {str(e)}")
            return ""

    def __parse_qwen(self, response_text, model_name):
        model_params = defaultdict(dict)
        for line in response_text.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            # 统一处理中英文冒号
            parts = re.split(r'[:：]', line, 3)
            if len(parts) >= 4:
                param, cond, val, unit = (p.strip() for p in parts)
                final_val = f"{val} {unit}" if unit not in ("NA", "") else val
                model_params[param][cond] = final_val
                self.global_conditions[param].add(cond)
        return model_params

    def __parse_deepseek(self, response_text, model_name):
        model_params = defaultdict(dict)
        for line in response_text.split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            # 适配GPT可能使用的不同分隔符
            parts = re.split(r'\s*:\s*', line, 3)
            if len(parts) >= 4:
                param, cond, val, unit = (p.strip() for p in parts)
                final_val = f"{val} {unit}" if unit.lower() not in ("na", "n/a") else val
                model_params[param][cond] = final_val
                self.global_conditions[param].add(cond)
        return model_params

    def __generate_headers(self):
        headers = ["Model"]
        for param in self.params:
            if param in self.global_conditions:
                sorted_conditions = sorted(
                    self.global_conditions[param],
                    key=lambda x: next((
                        idx for model_data in self.all_models_data
                        for idx, cond in enumerate(model_data['params'].get(param, {})) if cond == x
                    ), float('inf'))
                )
                for cond in sorted_conditions:
                    headers.append(f"{param} [{cond}]")
        return headers

    def process_all_models(self):
        for model in self.MODELS:
            print(f"\n正在通过 [{model['name']}] 查询参数...")
            model_type = model['type']

            response_text = self.model_run[model_type](model['name'])
            print(f"[{model['name']}] 原始响应：\n{response_text or '无响应'}")

            # 解析响应
            if response_text and model_type in self.string2csv:
                parser = self.string2csv[model_type]
                model_params = parser(response_text, model['name'])
                
                self.all_models_data.append({
                    "name": model['name'],
                    "params": model_params
                })

        # 生成最终表头
        self.headers = self.__generate_headers()
        return self.headers, self.all_models_data, self.params, self.global_conditions



processor = MD2CSV(md_content, MODELS, params, allapi)
headers, all_models_data, params, global_conditions = processor.process_all_models()

md2csv = MD2CSV(md_content, MODELS, params, allapi)

# 调用处理方法，获取Qwen和Deepseek的DataFrame
qwen_df, deepseek_df = md2csv.process_all_models()

# 合并 DataFrame（按化合物名称进行合并，使用 outer 合并方式确保所有数据都被保留）
merged_df = pd.merge(qwen_df, deepseek_df, on='化合物名称', how='outer', suffixes=('_x', '_y'))
merged_df.drop('Model', axis=1, inplace=True)
# 去除重复列，并选择保留一个列
# 这里可以通过自定义逻辑，选择要保留的列，去除 `_x` 和 `_y` 后缀
for col in merged_df.columns:
    if '_x' in col:
        # 获取对应的 _y 列
        corresponding_col_y = col.replace('_x', '_y')
        if corresponding_col_y in merged_df.columns:
            # 选择保留 _x 列，移除 _y 列
            merged_df[col.replace('_x', '')] = merged_df[col]  # 保留 _x 列
            merged_df.drop([col, corresponding_col_y], axis=1, inplace=True)

# 打印合并后的 DataFrame
print(merged_df)

merged_path = "/home/stu/yc/decimer/csv_folder/merged_output.csv"


# 检查文件夹是否存在，如果不存在则创建
folder = os.path.dirname(merged_path)
if not os.path.exists(folder):
    os.makedirs(folder)

# 保存文件

merged_df.to_csv(merged_path, index=False, encoding="utf-8")

merged_path = "/home/stu/yc/decimer/csv_folder/merged_output.csv"


# 检查文件夹是否存在，如果不存在则创建
folder = os.path.dirname(merged_path)
if not os.path.exists(folder):
    os.makedirs(folder)

# 保存文件

merged_df.to_csv(merged_path, index=False, encoding="utf-8")

class PngFind:
    def __init__(self, allapi, raw_name, image_folder):
        self.allapi = allapi
        self.raw_name = raw_name
        self.image_folder = image_folder

    # --------------------------
    # 文件处理模块
    # --------------------------
    def natural_sort_key(self, s):
        """生成自然排序键"""
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)
        ]
    
    def extract_numbers(self, filename):
        """增强版数字提取，优先结构化匹配"""
        # 优先匹配 page_X_img_Y 格式
        structured_match = re.findall(r'(?:page|p)[_-]?(\d+).*?(?:img|image)[_-]?(\d+)', filename, re.IGNORECASE)
        if structured_match:
            return list(structured_match[0])
        
        # 次选所有连续数字
        return re.findall(r'\d+', filename)
    
    def get_page_subpage(self, filename):
        """获取页码和子页号（子编号从1开始）"""
        numbers = self.extract_numbers(filename)
        
        # 处理逻辑
        if len(numbers) >= 2:
            return (int(numbers[0]), int(numbers[1]))
        elif numbers:
            return (int(numbers[0]), 1)  # 单个数字时子编号为1
        return (0, 1)  # 无数字文件默认0-1
    
    def process_image_files(self):
        """处理图片文件并生成映射关系"""
        image_files = [f for f in os.listdir(self.image_folder) 
                      if f.lower().endswith(('.png', '.jpeg', '.jpg'))]
        
        # 四重排序保障
        image_files.sort(key=lambda x: (
            self.get_page_subpage(x)[0],   # 主页码
            self.get_page_subpage(x)[1],   # 子页码
            self.natural_sort_key(x),      # 自然排序
            x.lower()                      # 字母排序
        ))
        
        # 生成映射表
        number_mapping = []
        page_order = []
        for f in image_files:
            main, sub = self.get_page_subpage(f)
            number_mapping.append(f"{main}-{sub}")
            page_order.append(main)

        return image_files, number_mapping, page_order

    # --------------------------
    # API处理模块
    # --------------------------
    def call_dashscope_api(self, messages):
        """调用多模态API"""
        response = dashscope.MultiModalConversation.call(
            api_key=self.allapi.get("qwen"),
            model="qwen-vl-max-latest",
            messages=messages
        )
        return response.output.choices[0].message.content[0]["text"]
    
    def process_api_response(self, out, max_index):
        """解析API响应内容"""
        # 分段处理
        sections = [s.strip() for s in out.split('\n') if s.strip()]
        index_section = sections[0] if sections else ""
        name_sections = sections[1:] if len(sections) > 1 else []
        
        # 解析索引
        indexes = []
        if index_section:
            indexes = [int(i) for i in index_section.split(',') if i.strip().isdigit()]
        
        # 解析名称
        names = []
        for sec in name_sections:
            names.extend([n.strip() for n in sec.split(',')])
        
        # 处理索引有效性
        valid_flags = []
        converted_indexes = []
        for idx in indexes:
            is_valid = 1 <= idx <= max_index
            valid_flags.append(is_valid)
            converted_indexes.append(idx-1 if is_valid else None)
        
        # 分子名称处理
        missing_count = len([n for n in names if "未标明" in n])
        missing_names = [f"未标明-{i+1:02d}" for i in range(missing_count)]
        final_names = missing_names + [n for n in names if "未标明" not in n]
    
        # 查找BN-TP
        try:
            locations = final_names.index(self.raw_name)
        except ValueError:
            locations = None
        
        return {
            "raw_indexes": indexes,
            "valid_flags": valid_flags,
            "converted_indexes": converted_indexes,
            "number_mapping": final_names,
            "locations": locations
        }

    # --------------------------
    # 主程序
    # --------------------------
    def final_output(self):
        # 处理图片文件
        sorted_files, number_codes, page_nums = self.process_image_files()
        max_index = len(sorted_files)  # 最大有效1-based索引
        
        # 构建消息
        messages = [
            {
                "role": "system",
                "content": [{"text": "你是一名化学专家，直接以数字形式告诉我第几张图片有化学结构？"}]
            },
            {
                "role": "user",
                "content": [{"image": os.path.join(self.image_folder, f)} for f in sorted_files] + 
                           [{"text": "有化学结构的图中有几个分子结构，请按照从左到右，从上到下的顺序将图片中的化学分子式名称列出，没有标注名称的分子用未标明-01、未标明-02等代替，把代替后和原本存在的分子名称输出成逗号分隔"}]
            }
        ]
        
        # 调用API
        api_response = self.call_dashscope_api(messages)
        
        # 解析响应
        result = self.process_api_response(api_response, max_index)

        # 生成最终输出
        final_output = {
            "所有原始索引": result["raw_indexes"],
            "有效索引": [i + 1 for i in result["converted_indexes"] if i is not None],
            "无效索引": [result["raw_indexes"][i] for i, flag in enumerate(result["valid_flags"]) if not flag],
            "对应编号": [number_codes[i] for i in result["converted_indexes"] if i is not None],
            "对应页码": [page_nums[i] for i in result["converted_indexes"] if i is not None],
            "分子列表": result["number_mapping"],
            "分子位置": result["locations"]
        }

        # 将所有结果写入文本文件
        output_file = "output_results.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        
        # 打印结果
        print("API原始响应:", api_response)
        print("文件总数:", len(sorted_files))
        print("所有原始索引:", final_output["所有原始索引"])
        print("有效图片索引:", final_output["有效索引"])
        print("无效图片索引:", final_output["无效索引"])
        print("对应图片编号:", final_output["对应编号"])
        print("对应主页码:", final_output["对应页码"])
        print("分子名称列表:", final_output["分子列表"])
        print("分子位置:", final_output["分子位置"])
        
        # 动态创建变量并赋值
        page_numbers = final_output["对应页码"]
        smiles_numbers = final_output["分子位置"]
        
        if not isinstance(smiles_numbers, list):
            smiles_numbers = [smiles_numbers]  # 如果是单个整数，将其转换为列表
        
        # 创建全局变量
        for idx, page_num in enumerate(page_numbers):
            variable_name = f"chemical{idx + 1:02d}"
            globals()[variable_name] = page_num  
        
        for idx, smiles_num in enumerate(smiles_numbers):
            variable_name = f"smiles{idx + 1:02d}"
            globals()[variable_name] = smiles_num
        
        # 打印结果来验证
        print("所有化学品变量:")
        for idx, page_num in enumerate(page_numbers):
            variable_name = f"chemical{idx + 1:02d}"
            print(f"{variable_name}: {globals()[variable_name]}")
        
        print("所有化学名称变量:")
        for idx, smiles_num in enumerate(smiles_numbers):
            variable_name = f"smiles{idx + 1:02d}"
            print(f"{variable_name}: {globals()[variable_name]}")
        
        print(f"所有数据已保存至 {output_file}")

png_find=PngFind(allapi,raw_name,image_folder)
final_output=png_find.final_output()

def extract_pdf_and_decode_smiles(file_note, segments, poppler_path=poppler_path, together=False):
    """
    从PDF中提取页面并解码化学结构为SMILES符号。

    file_note: dict, 包含PDF路径的字典
    segments: [(start, end)], 每个元素为元组，表示需要提取的页码范围
    poppler_path: str, Poppler路径，用于将PDF页面转换为图像（需要Poppler进行PDF转图像）
    together: bool, 是否将所有页面合并成一个PDF（默认为False）
    """
    # 提取PDF页面
    file_path = file_note.get("file_path")
    pdf_writer = PdfWriter()
    pdf_writer_segment = PdfWriter()
    p = Path(file_path)

    with open(file_path, 'rb') as read_stream:
        pdf_reader = PdfReader(read_stream)

        for segment in segments:
            start_page, end_page = segment  # 直接解包元组

            for page_num in range(start_page, end_page):
                if together:
                    pdf_writer.add_page(pdf_reader.pages[page_num])
                else:
                    pdf_writer_segment.add_page(pdf_reader.pages[page_num])

        # 输出提取的PDF页面
        if together:
            output = p.parent / p.with_stem(f'{p.stem}_extracted')
            with open(output, 'wb') as out:
                pdf_writer.write(out)
        else:
            for segment in segments:
                start_page, end_page = segment  # 直接解包元组
                output = p.parent / p.with_stem(f'{p.stem}_pages_{start_page}-{end_page}')
                with open(output, 'wb') as out:
                    pdf_writer_segment.write(out)

    # 使用Poppler将提取的页面转换为图像
    if poppler_path:
        pages = convert_from_path(str(file_path), 300, poppler_path=poppler_path)
        print(f"总共提取了 {len(pages)} 张图像")

        # 从提取的图像中重新组成新的segments
        image_segments = segment_chemical_structures(np.array(pages[1]), expand=True, visualization=True)

        # 从识别到的图像中获取第x个分子
        img = Image.fromarray(image_segments[smiles01-1])
        img.save("test.png")
        img.show()

        # 解码SMILES符号
        SMILES = predict_SMILES("test.png")
        print(f"🎉 解码得到的SMILES: {SMILES}")

        # 将SMILES转换为RDKit分子并显示结构
        mol = Chem.MolFromSmiles(SMILES)
        mol_img = Draw.MolToImage(mol)
        mol_img.show()
    return SMILES
# 示例调用函数
SMILES=extract_pdf_and_decode_smiles(file_note, [(chemical01,chemical02)], poppler_path=poppler_path, together=False)



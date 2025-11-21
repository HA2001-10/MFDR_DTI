import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoModel, AutoTokenizer
import pickle
import os
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import MACCSkeys
import warnings
from rdkit import RDLogger
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
RDLogger.DisableLog('rdApp.warning')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 全局变量
feature_df = None
scaler = None
feature_size = 164  # 确认的特征数量


def load_features_with_scaling(dataset_name):
    global feature_df, scaler
    if feature_df is None:
        # 动态构建CSV文件路径
        csv_file_path = f'/public/home/xiedan/MFMC_DTI/DataSets/{dataset_name}_all_features.csv'
        print(f"Loading features from: {csv_file_path}")
        feature_df = pd.read_csv(csv_file_path)
        # 保存原始索引
        protein_ids = feature_df['ProteinID']
        # 标准化特征
        feature_values = feature_df.drop('ProteinID', axis=1).values
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_values)
        # 重新创建数据框
        feature_df = pd.DataFrame(scaled_features,
                                  index=protein_ids,
                                  columns=feature_df.columns[1:])
    return feature_df


# Bert预训练模型配置
prot_model_path = "/public/home/xiedan/MFMC_DTI/huggingface/prot_bert_bfd"
chem_model_path = "/public/home/xiedan/MFMC_DTI/huggingface/PubChem10M_SMILES_BPE_450k"

# 加载蛋白质BERT模型及其分词器
prot_tokenizer = AutoTokenizer.from_pretrained(
    prot_model_path,
    do_lower_case=False,
    local_files_only=True
)
prot_model = AutoModel.from_pretrained(
    prot_model_path,
    local_files_only=True
).to(DEVICE)

# 加载化合物BERT模型及其分词器
chem_tokenizer = AutoTokenizer.from_pretrained(
    chem_model_path,
    do_lower_case=False,
    local_files_only=True
)
chem_model = AutoModel.from_pretrained(
    chem_model_path,
    local_files_only=True
).to(DEVICE)

# Bert特征缓存字典
prot_bert_features = {}
chem_bert_features = {}


def extract_protein_bert_features(sequence):
    """提取蛋白质Bert特征"""
    if sequence in prot_bert_features:
        return prot_bert_features[sequence]

    if len(sequence) > 5000:
        sequence = sequence[0:5000]

    protein_input = prot_tokenizer.batch_encode_plus(
        [" ".join(sequence)],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=1024
    )

    p_IDS = torch.tensor(protein_input["input_ids"]).to(DEVICE)
    p_a_m = torch.tensor(protein_input["attention_mask"]).to(DEVICE)

    with torch.no_grad():
        prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)

    prot_feature = prot_outputs.last_hidden_state.squeeze(0).mean(dim=0).to('cpu').data.numpy()
    # 确保特征是一维的
    if len(prot_feature.shape) > 1:
        prot_feature = prot_feature.flatten()
    prot_bert_features[sequence] = prot_feature
    return prot_feature


def extract_compound_bert_features(smiles):
    """提取化合物Bert特征"""
    if smiles in chem_bert_features:
        return chem_bert_features[smiles]

    if len(smiles) > 512:
        smiles = smiles[0:512]

    chem_input = chem_tokenizer.batch_encode_plus(
        [smiles],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512
    )

    c_IDS = torch.tensor(chem_input["input_ids"]).to(DEVICE)
    c_a_m = torch.tensor(chem_input["attention_mask"]).to(DEVICE)

    with torch.no_grad():
        chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)

    chem_feature = chem_outputs.last_hidden_state.squeeze(0).mean(dim=0).to('cpu').data.numpy()
    # 确保特征是一维的
    if len(chem_feature.shape) > 1:
        chem_feature = chem_feature.flatten()
    chem_bert_features[smiles] = chem_feature
    return chem_feature


def save_bert_features():
    """保存Bert特征到文件"""
    with open("prot_bert_features.pkl", "wb") as f:
        pickle.dump(prot_bert_features, f)
    with open("chem_bert_features.pkl", "wb") as f:
        pickle.dump(chem_bert_features, f)


def load_bert_features():
    """从文件加载Bert特征"""
    global prot_bert_features, chem_bert_features
    if os.path.exists("prot_bert_features.pkl"):
        with open("prot_bert_features.pkl", "rb") as f:
            prot_bert_features = pickle.load(f)
    if os.path.exists("chem_bert_features.pkl"):
        with open("chem_bert_features.pkl", "rb") as f:
            chem_bert_features = pickle.load(f)



CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25

# 指纹参数
MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


# 指纹提取函数 - 更新为使用 MorganGenerator
def morgan_binary_features_generator(mol: Chem.Mol) -> np.ndarray:
    """Generates Morgan fingerprint for a molecule using MorganGenerator."""
    try:
        # 使用新的 MorganGenerator API
        fp_generator = AllChem.MorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_NUM_BITS)
        features_vec = fp_generator.GetFingerprint(mol)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features
    except AttributeError:
        # 如果 RDKit 版本较旧，回退到旧方法
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NUM_BITS)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
        return features


def Avalon_features_generator(mol: Chem.Mol) -> np.ndarray:
    """Generates Avalon fingerprint for a molecule."""
    features_vec = pyAvalonTools.GetAvalonFP(mol)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features


def MACCS_features_generator(mol: Chem.Mol) -> np.ndarray:
    """Generates MACCS Keys fingerprint for a molecule."""
    features_vec = MACCSkeys.GenMACCSKeys(mol)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, dtype=np.int64)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


class CustomDataSet(Dataset):
    def __init__(self, pairs, dataset_name):
        self.pairs = pairs
        self.dataset_name = dataset_name  # 保存数据集名称
        self._preload_bert_features()
        # 预先计算所有分子的指纹
        self.fingerprints = {}
        for pair in tqdm(pairs, desc="Precomputing fingerprints"):
            compoundstr = pair.strip().split()[-3]
            mol = Chem.MolFromSmiles(compoundstr)
            if mol:
                try:
                    morgan = morgan_binary_features_generator(mol)
                    avalon = Avalon_features_generator(mol)
                    maccs = MACCS_features_generator(mol)
                    self.fingerprints[compoundstr] = {
                        'morgan': torch.FloatTensor(morgan),
                        'avalon': torch.FloatTensor(avalon),
                        'maccs': torch.FloatTensor(maccs)
                    }
                except Exception as e:
                    print(f"Error generating fingerprints for {compoundstr}: {e}")
                    # 创建零向量作为备用
                    dummy_fp = np.zeros(MORGAN_NUM_BITS)
                    self.fingerprints[compoundstr] = {
                        'morgan': torch.FloatTensor(dummy_fp),
                        'avalon': torch.FloatTensor(dummy_fp),
                        'maccs': torch.FloatTensor(dummy_fp)
                    }
            else:
                # 如果分子无效，创建零向量
                dummy_fp = np.zeros(MORGAN_NUM_BITS)
                self.fingerprints[compoundstr] = {
                    'morgan': torch.FloatTensor(dummy_fp),
                    'avalon': torch.FloatTensor(dummy_fp),
                    'maccs': torch.FloatTensor(dummy_fp)
                }

    def _preload_bert_features(self):
        """预加载所有数据的Bert特征"""
        print("预加载Bert特征...")
        for pair in tqdm(self.pairs):
            pair_data = pair.strip().split()
            if len(pair_data) >= 3:
                compoundstr, proteinstr = pair_data[-3], pair_data[-2]
                # 提取并缓存特征
                extract_compound_bert_features(compoundstr)
                extract_protein_bert_features(proteinstr)

    def __getitem__(self, item):
        pair = self.pairs[item]
        pair_data = pair.strip().split()
        compoundstr, proteinstr, label = pair_data[-3], pair_data[-2], pair_data[-1]
        drug_id, protein_id = pair_data[0], pair_data[1]
        protein_key = f"{drug_id}_{protein_id}"

        # 传统特征
        compoundint = label_smiles(compoundstr, CHARISOSMISET, 100)
        proteinint = label_sequence(proteinstr, CHARPROTSET, 1000)
        label = float(label)

        # 获取指纹特征
        fingerprints = self.fingerprints.get(compoundstr, None)
        if fingerprints is None:
            # 如果指纹计算失败，创建零向量
            dummy_fp = np.zeros(MORGAN_NUM_BITS)
            fingerprints = {
                'morgan': torch.FloatTensor(dummy_fp),
                'avalon': torch.FloatTensor(dummy_fp),
                'maccs': torch.FloatTensor(dummy_fp)
            }

        # 加载额外特征 - 修复这里：传递数据集名称
        feature_df = load_features_with_scaling(self.dataset_name)  # 添加 self.dataset_name
        if protein_key in feature_df.index:
            features_values = feature_df.loc[protein_key].values
            if len(features_values.shape) > 1:
                features_values = features_values[0]
            protein_features = torch.tensor(features_values.astype(np.float32))
        else:
            # 静默处理，不显示警告
            mean_features = torch.tensor(feature_df.mean().values.astype(np.float32))
            protein_features = mean_features

        compound_bert = extract_compound_bert_features(compoundstr)
        protein_bert = extract_protein_bert_features(proteinstr)

        return {
            'compound': compoundint,
            'protein': proteinint,
            'compound_bert': compound_bert,
            'protein_bert': protein_bert,
            'label': label,
            'morgan_fp': fingerprints['morgan'],
            'avalon_fp': fingerprints['avalon'],
            'maccs_fp': fingerprints['maccs'],
            'protein_features': protein_features
        }

    def __len__(self):
        return len(self.pairs)


def collate_fn(batch_data):
    N = len(batch_data)
    compound_max = 100
    protein_max = 1000

    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)

    # 指纹特征
    morgan_features = []
    avalon_features = []
    maccs_features = []

    # 检查是否包含Bert特征
    has_bert = 'compound_bert' in batch_data[0]

    bert_dim_compound = batch_data[0]['compound_bert'].shape[0]
    bert_dim_protein = batch_data[0]['protein_bert'].shape[0]

    compound_bert_new = torch.zeros((N, bert_dim_compound), dtype=torch.float32)
    protein_bert_new = torch.zeros((N, bert_dim_protein), dtype=torch.float32)

    # 获取额外特征维度
    feature_size = batch_data[0]['protein_features'].shape[0]
    protein_features_new = torch.zeros((N, feature_size), dtype=torch.float32)

    for i, data in enumerate(batch_data):
        compound_new[i] = torch.from_numpy(data['compound'])
        protein_new[i] = torch.from_numpy(data['protein'])
        labels_new[i] = np.int64(data['label'])

        # 收集指纹特征
        morgan_features.append(data['morgan_fp'])
        avalon_features.append(data['avalon_fp'])
        maccs_features.append(data['maccs_fp'])

        # 收集额外特征
        protein_features_new[i] = data['protein_features']

        compound_bert_new[i] = torch.from_numpy(data['compound_bert'])
        protein_bert_new[i] = torch.from_numpy(data['protein_bert'])

    # 堆叠指纹特征
    morgan_features = torch.stack(morgan_features)
    avalon_features = torch.stack(avalon_features)
    maccs_features = torch.stack(maccs_features)


    return (
        compound_new,
        protein_new,
        compound_bert_new,
        protein_bert_new,
        labels_new,
        morgan_features,
        avalon_features,
        maccs_features,
        protein_features_new
    )
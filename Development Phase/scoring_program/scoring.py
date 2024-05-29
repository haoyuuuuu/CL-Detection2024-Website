import numpy as np
import pandas as pd
import argparse
import os
import json

def calculate_prediction_metrics(result_dict: dict):
    """
    function to calculate prediction metrics | 计算评价指标
    :param result_dict: a dict, which stores every image's predict result and its ground truth landmark
                        一个字典，存储每个图像的预测结果及其真实地标
    :return: MRE and 2mm SDR metrics | MRE 和 2mm SDR 指标
    """
    n_landmarks = 0
    sdr_landmarks = 0
    n_landmarks_error = 0
    for file_path, landmark_dict in result_dict.items():
        spacing = landmark_dict['spacing']
        landmarks, predict_landmarks = landmark_dict['gt'], landmark_dict['predict']

        # landmarks number
        n_landmarks = n_landmarks + np.shape(landmarks)[0]

        # mean radius error (MRE)
        each_landmark_error = np.sqrt(np.sum(np.square(landmarks - predict_landmarks), axis=1)) * spacing
        n_landmarks_error = n_landmarks_error + np.sum(each_landmark_error)

        # 2mm success detection rate (SDR)
        sdr_landmarks = sdr_landmarks + np.sum(each_landmark_error < 2)

    mean_radius_error = n_landmarks_error / n_landmarks
    sdr = sdr_landmarks / n_landmarks
    print('Mean Radius Error (MRE): {}, 2mm Success Detection Rate (SDR): {}'.format(mean_radius_error, sdr))
    return mean_radius_error, sdr

def main(config):

    # CSV文件路径
    gt_landmarks_path = os.path.join(config.input_dir_path, 'ref', 'labels.csv')
    pred_landmarks_path = os.path.join(config.input_dir_path, 'res', 'predicts.csv')

    gt_df = pd.read_csv(gt_landmarks_path)
    pred_df = pd.read_csv(pred_landmarks_path)

    # test result dict | 测试结果字典
    test_result_dict = {}

    for index, row in gt_df.iterrows():
        image_file, spacing = str(gt_df.iloc[index, 0]), float(gt_df.iloc[index, 1])

        gt_landmarks = gt_df.iloc[index, 2:].values.astype('float')
        gt_landmarks = gt_landmarks.reshape(-1, 2)

        pred_landmarks = pred_df[pred_df.iloc[:, 0] == image_file].iloc[0, 1:].values.astype('float')
        pred_landmarks = pred_landmarks.reshape(-1, 2)

        test_result_dict[image_file] = {'spacing': spacing,
                                        'gt': np.asarray(gt_landmarks),
                                        'predict': np.asarray(pred_landmarks)}

    # calculate prediction metrics | 计算预测指标
    MRE,SDR = calculate_prediction_metrics(test_result_dict)
    score_dict = {'MRE':MRE, 'SDR':SDR}

    save_json_path = os.path.join(config.output_dir_path,'scores.json')

    with open(save_json_path, 'w', encoding='utf-8') as file:
        json.dump(score_dict, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data parameters | 数据参数
    # Path Settings | 路径设置
    parser.add_argument('--input_dir_path', type=str, default='/app/input/')
    parser.add_argument('--output_dir_path', type=str, default='/app/output/')

    experiment_config = parser.parse_args()
    main(experiment_config)




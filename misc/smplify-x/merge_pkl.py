import argparse
import pickle
import os
import numpy as np

parser = argparse.ArgumentParser(description='Merge pickle files')
parser.add_argument('--input_pkl')
parser.add_argument('--fits_dir')
parser.add_argument('--output_pkl')


def merge_pkl(input_pkl, fits_dir, output_pkl):
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)
    for i in range(len(data)):
        datum = data[i]
        poses = []
        betas = []
        has_smpl = []
        for j in range(len(datum['bboxes'])):
            fit_fname = os.path.join(fits_dir, 'results', str(i).zfill(7), str(j).zfill(2) + '.pkl')
            try:
                fit_data = pickle.load(open(fit_fname, 'rb'))
                beta = fit_fname['betas']
                pose = np.concatenate((fit_fname['global_orient'], fit_fname['body_pose']), axis=-1)
                has_smpl_ = np.ones((1,))
            except:
                pose = np.zeros((1, 72))
                beta = np.zeros((1, 10))
                has_smpl_ = np.zeros((1,))
                pass
            poses.append(pose)
            betas.append(beta)
            has_smpl.append(has_smpl_)
        pose = np.concatenate(poses, axis=0)
        betas = np.concatenate(betas, axis=0)
        datum['betas'] = betas
        datum['pose'] = pose
        datum['has_smpl'] = has_smpl

    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    args = parser.parse_args()
    merge_pkl(args.input_pkl, args.fits_dir, args.output_pkl)

import os
import numpy as np
from dipy.io.streamline import load_trk
from dipy.tracking.streamline import transform_streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from challenge_scoring.io.results import save_results
from challenge_scoring.metrics.scoring import score_submission
from challenge_scoring.utils.attributes import load_attribs
from challenge_scoring.utils.filenames import mkdir

import pickle


def get_rotmat(z_angle):
    res = np.array([[np.cos(z_angle), np.sin(z_angle), 0, 0],
                    [-np.sin(z_angle), np.cos(z_angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    return res


def get_transmat(coords):
    res = np.eye(4)
    res[0:3, -1] = coords

    return res


def main(rootdir, trkfile, reffile, basedir, gtbundl):
    bundl_attribs = load_attribs(os.path.join(basedir, gtbundl))

    trackts = {}
    reffiles = {}
    scoredict = {}

    for _file in os.listdir(rootdir):
        if _file.startswith('angle_'):
            print(f'loading trackt in folder {_file}')
            trackts[_file] = load_trk(os.path.join(rootdir, _file, trkfile), 'same')
            reffiles[_file] = os.path.join(rootdir, _file, reffile)
            # break  # TODO remove

    # sort the keys
    t_keys = list(trackts.keys())
    t_keys.sort(key=lambda x: int(x.split('_')[-1]))

    angles = [int(x.split('_')[-1]) for x in t_keys]
    transvec = np.array([96, 96, 0.])

    for angle, t_key in zip(angles, t_keys):

        trackt = trackts[t_key]
        ref_dir = reffiles[t_key]

        lines_moved = trackt.streamlines
        lines_moved = transform_streamlines(lines_moved, get_transmat(transvec))
        lines_moved = transform_streamlines(lines_moved,
                                            get_rotmat(np.radians(-angle)))
        lines_moved = transform_streamlines(lines_moved, get_transmat(-transvec))

        tract_moved = StatefulTractogram(lines_moved, ref_dir, Space.RASMM)
        tract_file = os.path.join(rootdir, t_key, 'tractogram_mooved.trk')
        save_tractogram(tract_moved, tract_file, bbox_valid_check=False)

        # scores = score_submission(
        #     streamlines_fname=tract_file,
        #     tracts_attribs={'orientation': 'unknown'},
        #     base_data_dir=basedir,
        #     basic_bundles_attribs=bundl_attribs)

        scores = score_submission(
            streamlines_fname=tract_file,
            base_data_dir=basedir,
            basic_bundles_attribs=bundl_attribs)
        scoredict[t_key] = scores

    pickle.dump(scoredict, open('/fabi_project/data/scratch/NOW/so3_coredict_5deg', 'wb'))
    print('DONE')


if __name__ == '__main__':
    # _rootdir = '/fabi_project/data/ttl_anat_priors/fabi_tests/experiments/SACAutoFiberCupTrain/2022-10-11-10_18_50/1111/test_out2'
    # _trkfile = 'tractogram_SACAutoFiberCupTrain_2022-10-11-10_18_50_fibercup.trk'
    _rootdir = '/fabi_project/data/ttl_anat_priors/fabi_tests/experiments/SACSo3FiberCupTrain/2022-11-09-13_06_33/1111/test_out/'
    _trkfile = 'tractogram_SACSo3FiberCupTrain_2022-11-09-13_06_33_fibercup.trk'

    _reffile = 'fibercup_wm.nii.gz'
    _basedir = '/fabi_project/data/ttl_anat_priors/raw/fibercup/scoring_data'
    _gtbundl = 'gt_bundles_attributes.json'


    main(_rootdir, _trkfile, _reffile, _basedir, _gtbundl)

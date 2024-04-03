import argparse
import os


def is_available(pdbid):
    assert os.path.exists('./dataset/16sys_db/all_md.list'), 'all_md.list does not exist'
    with open('./dataset/16sys_db/all_md.list', 'r') as f:
        available_list = f.readlines()
    available_list = [x.strip() for x in available_list]
    if pdbid in available_list:
        return True
    else:
        return False


def main(selected_pdbid, num_samples):
    assert is_available(selected_pdbid), 'selected pdbid is not available'
    with open('./dataset/16sys_db/test_md.list', 'w') as f:
        for i in range(num_samples):
            f.write(selected_pdbid+'\n')


if __name__ == "__main__":
    # add a output folder argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_pdbid', type=str)
    parser.add_argument('--num_samples', type=int, default=50)
    args = parser.parse_args()
    main(args.selected_pdbid, args.num_samples)

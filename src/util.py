from timeit import default_timer
import argparse
import os
import shutil


def pretty_evaluation(results):
    return f"""
---------------------------------
AP Retweet:                {results['retweet_AP']:.4f}
RCE Retweet:               {results['retweet_RCE']:.4f}
---------------------------------
AP Reply:                  {results['reply_AP']:.4f}
RCE Reply:                 {results['reply_RCE']:.4f}
---------------------------------
AP Like:                   {results['like_AP']:.4f}
RCE Like:                  {results['like_RCE']:.4f}
---------------------------------
AP RT with comment:        {results['retweet_with_comment_AP']:.4f}
RCE RT with comment:       {results['retweet_with_comment_RCE']:.4f}
""".strip()


class Stage(object):
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.start = default_timer()
        print(self.msg, flush=True)
        return self

    def __exit__(self, *args):
        t = default_timer() - self.start
        print(f"Done {t:.2f} seconds", flush=True)
        print()
        self.time = t


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def rm_dir_contents(path):
    for filename in os.listdir(path):
        if filename.startswith("."):
            continue  # skip dotfiles

        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Error: %s' % (file_path, e))
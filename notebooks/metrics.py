from sklearn.metrics import average_precision_score, log_loss


def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def relative_cross_entropy_score(gt, pred):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

def compute_score(y_true, y_score):
    ap = average_precision_score(y_true, y_score)
    rce = relative_cross_entropy_score(y_true, y_score)
    return ap, rce

def pretty_evaluation(results):
    mAP = (
        results['retweet_AP']
        +results['reply_AP']
        +results['like_AP']
        +results['retweet_with_comment_AP']
    ) / 4
    mRCE = (
        results['retweet_RCE']
        +results['reply_RCE']
        +results['like_RCE']
        +results['retweet_with_comment_RCE']
    ) / 4
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

---------------------------------

mAP                        {mAP:.4f}
mRCE                       {mRCE:.4f}

""".strip()

def just_mAP_mRCE(results):
    mAP = (
        results['retweet_AP']
        +results['reply_AP']
        +results['like_AP']
        +results['retweet_with_comment_AP']
    ) / 4
    mRCE = (
        results['retweet_RCE']
        +results['reply_RCE']
        +results['like_RCE']
        +results['retweet_with_comment_RCE']
    ) / 4
    return f"""
mAP                        {mAP:.4f}
mRCE                       {mRCE:.4f}

""".strip()
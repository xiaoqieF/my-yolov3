import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ap_per_class(tp, conf, pred_cls, target_cls, plot=True, save_dir='./run', names=()):
    """用于val.py中计算每个类的mAP
    计算每一个类的AP指标(average precision)还可以 绘制P-R曲线
    mAP基本概念: https://www.bilibili.com/video/BV1ez4y1X7g2
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    :params tp(correct): [pred_sum, 10]=[1905, 10] bool 整个数据集所有图片中所有预测框在每一个iou条件下(0.5~0.95)10个是否是TP
    :params conf: [img_sum]=[1905] 整个数据集所有图片的所有预测框的conf
    :params pred_cls: [img_sum]=[1905] 整个数据集所有图片的所有预测框的类别
            这里的tp、conf、pred_cls是一一对应的
    :params target_cls: [gt_sum]=[929] 整个数据集所有图片的所有gt框的class
    :params plot: bool
    :params save_dir: runs\train\exp30
    :params names: dict{key(class_index):value(class_name)} 获取数据集所有类别的index和对应类名
    :return p[:, i]: [nc] 最大平均f1时每个类别的precision
    :return r[:, i]: [nc] 最大平均f1时每个类别的recall
    :return ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
    :return f1[:, i]: [nc] 最大平均f1时每个类别的f1
    :return unique_classes.astype('int32'): [nc] 返回数据集中所有的类别index
    """
    # 计算mAP 需要将tp按照conf降序排列
    # Sort by objectness  按conf从大到小排序 返回数据对应的索引
    i = np.argsort(-conf)
    # 得到重新排序后对应的 tp, conf, pre_cls
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes  对类别去重, 因为计算ap是对每类进行
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # 数据集类别数 number of classes

    # Create Precision-Recall curve and compute AP for each class
    # px: [0, 1] 中间间隔1000个点 x坐标(用于绘制P-Conf、R-Conf、F1-Conf)
    # py: y坐标[] 用于绘制IOU=0.5时的PR曲线
    px, py = np.linspace(0, 1, 1000), []  # for plotting

    # 初始化 对每一个类别在每一个IOU阈值下 计算AP P R   ap=[nc, 10]  p=[nc, 1000] r=[nc, 1000]
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):  # ci: index 0   c: class 0  unique_classes: 所有gt中不重复的class
        # i: 记录着所有预测框是否是c类别框   是c类对应位置为True, 否则为False
        i = pred_cls == c
        # n_l: gt框中的c类别框数量  = tp+fn   254
        n_l = (target_cls == c).sum()  # number of labels
        # n_p: 预测框中c类别的框数量   695
        n_p = i.sum()  # number of predictions

        # 如果没有预测到 或者 ground truth没有标注 则略过类别c
        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs(False Positive) and TPs(Ture Positive)   FP + TP = all_detections
            # tp[i] 可以根据i中的的True/False觉定是否删除这个数  所有tp中属于类c的预测框
            #       如: tp=[0,1,0,1] i=[True,False,False,True] b=tp[i]  => b=[0,1]
            # a.cumsum(0)  会按照对象进行累加操作
            # 一维按行累加如: a=[0,1,0,1]  b = a.cumsum(0) => b=[0,1,1,2]   而二维则按列累加
            # fpc: 类别为c 顺序按置信度排列 截至到每一个预测框的各个iou阈值下FP个数 最后一行表示c类在该iou阈值下所有FP数
            # tpc: 类别为c 顺序按置信度排列 截至到每一个预测框的各个iou阈值下TP个数 最后一行表示c类在该iou阈值下所有TP数
            fpc = (1 - tp[i]).cumsum(0)  # fp[i] = 1 - tp[i]
            tpc = tp[i].cumsum(0)

            # Recall=TP/(TP+FN)  加一个1e-16的目的是防止分母为0
            # n_l=TP+FN=num_gt: c类的gt个数=预测是c类而且预测正确+预测不是c类但是预测错误
            # recall: 类别为c 顺序按置信度排列 截至每一个预测框的各个iou阈值下的召回率
            recall = tpc / (n_l + 1e-16)  # recall curve  用于计算mAP
            # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的recall值  r=[nc, 1000]  每一行从小到大
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # 用于绘制R-Confidence(R_curve.png)

            # Precision=TP/(TP+FP)
            # precision: 类别为c 顺序按置信度排列 截至每一个预测框的各个iou阈值下的精确率
            precision = tpc / (tpc + fpc)  # precision curve  用于计算mAP
            # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的precision值  p=[nc, 1000]
            # 总体上是从小到大 但是细节上有点起伏 如: 0.91503 0.91558 0.90968 0.91026 0.90446 0.90506
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # 用于绘制P-Confidence(P_curve.png)

            # AP from recall-precision curve
            # 对c类别, 分别计算每一个iou阈值(0.5~0.95 10个)下的mAP
            for j in range(tp.shape[1]):  # tp [pred_sum, 10]
                # 这里执行10次计算ci这个类别在所有mAP阈值下的平均mAP  ap[nc, 10]
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # py: 用于绘制每一个类别IOU=0.5时的PR曲线

    # 计算F1分数 P和R的调和平均值  综合评价指标
    # 我们希望的是P和R两个越大越好, 但是P和R常常是两个冲突的变量, 经常是P越大R越小, 或者R越大P越小 所以我们引入F1综合指标
    # 不同任务的重点不一样, 有些任务希望P越大越好, 有些任务希望R越大越好, 有些任务希望两者都大, 这时候就看F1这个综合指标了
    # 返回所有类别, 横坐标为conf(值为px=[0, 1, 1000] 0~1 1000个点)对应的f1值  f1=[nc, 1000]
    f1 = 2 * p * r / (p + r + 1e-16)   # 用于绘制P-Confidence(F1_curve.png)

    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)                # 画pr曲线
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')       # 画F1_conf曲线
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')  # 画P_conf曲线
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')     # 画R_conf曲线

    # f1=[nc, 1000]   f1.mean(0)=[1000]求出所有类别在x轴每个conf点上的平均f1
    # .argmax(): 求出每个点平均f1中最大的f1对应conf点的index
    i = f1.mean(0).argmax()  # max F1 index

    # p=[nc, 1000] 每个类别在x轴每个conf值对应的precision
    # p[:, i]: [nc] 最大平均f1时每个类别的precision
    # r[:, i]: [nc] 最大平均f1时每个类别的recall
    # f1[:, i]: [nc] 最大平均f1时每个类别的f1
    # ap: [71, 10] 数据集每个类别在10个iou阈值下的mAP
    # unique_classes.astype('int32'): [nc] 返回数据集中所有的类别index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

def compute_ap(recall, precision):
    """用于ap_per_class函数中
    计算某个类别在某个iou阈值下的mAP
    Compute the average precision, given the recall and precision curves
    :params recall: (list) [1635] 在某个iou阈值下某个类别所有的预测框的recall  从小到大
                    (每个预测框的recall都是截至到这个预测框为止的总recall)
    :params precision: (list) [1635] 在某个iou阈值下某个类别所有的预测框的precision
                       总体上是从大到小 但是细节上有点起伏 如: 0.91503 0.91558 0.90968 0.91026 0.90446 0.90506
                       (每个预测框的precision都是截至到这个预测框为止的总precision)
    :return ap: Average precision 返回某类别在某个iou下的mAP(均值) [1]
    :return mpre: precision curve [1637] 返回 开头 + 输入precision(排序后) + 末尾
    :return mrec: recall curve [1637] 返回 开头 + 输入recall + 末尾
    """
    # 在开头和末尾添加保护值 防止全零的情况出现 value Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))  # [1637]
    mpre = np.concatenate(([1.], precision, [0.]))  # [1637]

    # Compute the precision envelope  np.flip翻转顺序
    # np.flip(mpre): 把一维数组每个元素的顺序进行翻转 第一个翻转成为最后一个
    # np.maximum.accumulate(np.flip(mpre)): 计算数组(或数组的特定轴)的累积最大值 令mpre是单调的 从小到大
    # np.flip(np.maximum.accumulate(np.flip(mpre))): 从大到小
    # 到这大概看明白了这步的目的: 要保证mpre是从大到小单调的(左右可以相同)
    # 我觉得这样可能是为了更好计算mAP 因为如果一直起起伏伏太难算了(x间隔很小就是一个矩形) 而且这样做误差也不会很大 两个之间的数都是间隔很小的
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':  # 用一些典型的间断点来计算AP
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)  [0, 0.01, ..., 1]
        #  np.trapz(list,list) 计算两个list对应点与点之间四边形的面积 以定积分形式估算AP 第一个参数是y 第二个参数是x
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'  # 采用连续的方法计算AP
        # 通过错位的方式 判断哪个点当前位置到下一个位置值发生改变 并通过！=判断 返回一个布尔数组
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        # 值改变了就求出当前矩阵的面积  值没变就说明当前矩阵和下一个矩阵的高相等所有可以合并计算
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    """用于ap_per_class函数
    Precision-recall curve  绘制PR曲线
    :params px: [1000] 横坐标 recall 值为0~1直接取1000个数
    :params py: list{nc} nc个[1000] 所有类别在IOU=0.5,横坐标为px(recall)时的precision
    :params ap: [nc, 10] 所有类别在每个IOU阈值下的平均mAP
    :params save_dir: runs\test\exp54\PR_curve.png  PR曲线存储位置
    :params names: {dict:80} 数据集所有类别的字典 key:value
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)  # 设置画布
    py = np.stack(py, axis=1)  # [1000, nc]

    # 画出所有类别在10个IOU阈值下的PR曲线
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):  # 如果<21 classes就一个个类画 因为要显示图例就必须一个个画
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:  # 如果>=21 classes 显示图例就会很乱 所以就不显示图例了 可以直接输入数组 x[1000] y[1000, 71]
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    # 画出所有类别在IOU=0.5阈值下的平均PR曲线
    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')     # 设置x轴标签
    ax.set_ylabel('Precision')  # 设置y轴标签
    ax.set_xlim(0, 1)           # x=[0, 1]
    ax.set_ylim(0, 1)           # y=[0, 1]
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")   # 显示图例
    fig.savefig(Path(save_dir), dpi=250)                     # 保存PR_curve.png图片

def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    """用于ap_per_class函数
    Metric-Confidence curve 可用于绘制 F1-Confidence/P-Confidence/R-Confidence曲线
    :params px: [0, 1, 1000] 横坐标 0-1 1000个点 conf   [1000]
    :params py: 对每个类, 针对横坐标为conf=[0, 1, 1000] 对应的f1/p/r值 纵坐标 [71, 1000]
    :params save_dir: 图片保存地址
    :parmas names: 数据集names
    :params xlabel: x轴标签
    :params ylabel: y轴标签
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)  # 设置画布

    # 画出所有类别的F1-Confidence/P-Confidence/R-Confidence曲线
    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):  # 如果<21 classes就一个个类画 因为要显示图例就必须一个个画
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:  # 如果>=21 classes 显示图例就会很乱 所以就不显示图例了 可以直接输入数组 x[1000] y[1000, 71]
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    # 画出所有类别在每个x点(conf)对应的均值F1-Confidence/P-Confidence/R-Confidence曲线
    y = py.mean(0)  # [1000] 求出所以类别在每个x点(conf)的平均值
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)  # 设置x轴标签
    ax.set_ylabel(ylabel)  # 设置y轴标签
    ax.set_xlim(0, 1)      # x=[0, 1]
    ax.set_ylim(0, 1)      # y=[0, 1]
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")  # 显示图例
    fig.savefig(Path(save_dir), dpi=250)                    # 保存png图片

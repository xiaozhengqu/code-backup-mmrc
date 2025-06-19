import numpy as np
import DataLoad
import Performance_Cal
import FuzzyART_Plot

'''
sa_art的重复多轮迭代版本,最后没有去除消亡的类簇（在最后一步去不去不重要，因为我们在计算performance时会去掉，所以去不去都不影响计算指标）

'''


def si_art_repeat_quick_stop_without_draw_without_AMR(M, label, rho, lam, alpha, expand_ratio, max_repeat_num, cluster_want_to_show=-1):
    """
    @param M: numpy arrary; m*n 特征矩阵; m是实例个数 ，n是特征数
    @param label: 维度：m，代表样本所属真实类别（从0开始）
    @param rho: 警戒参数(0-1)
    @param delta: 很小的数值，用于CM-ART策略中微微收缩边界
    @param beta: # has no significant impact on performance with a moderate value of [0.4,0.7]
    @param alpha: 避免除以0
    @return:
    """

    NAME = 'si_art'
    # print(NAME + "算法开始------------------")
    # -----------------------------------------------------------------------------------------------------------------------
    # Input parameters
    # no need to tune; used in choice function;
    # to avoid too small cluster weights (resulted by the learning method of ART; should be addressed sometime);
    # give priority to choosing denser clusters
    alpha = alpha

    # rho needs carefully tune; used to shape the inter-cluster similarity;
    # rho_v = 0.7 indicates an object will not be clustered to a cluster with visual similarity lower than 0.7
    # rho = 0.6

    # -----------------------------------------------------------------------------------------------------------------------
    # Initialization

    # complement coding
    M = np.concatenate([M, 1 - M], 1)

    # get data sizes
    row, col = M.shape

    # 接收每轮的结果情况
    performance_dic = {}
    # 保表示当前重复执行的轮数
    now_repeat_num = 0

    # -----------------------------------------------------------------------------------------------------------------------
    # Clustering process

    # print(NAME + "第一轮算法开始---------------------")
    now_repeat_num += 1

    # 用第一个样本初始化第一个cluster
    # Wv存放cluster权重参数，row行col列
    # 如果后面cluster小于row个，则把多余的行去掉
    Wv = np.zeros((row*2, col))

    # 记录各个类簇的每个特征的 频数（不是频率），row: clusters, col:features
    # 认为一个特征>0则该特征频数+1
    feature_salience_count = np.zeros((row*2, col))

    # 记录各个类簇的每个特征的 均值，row: clusters, col:features
    feature_mean = np.zeros((row*2, col))

    # 计算 方差 过程中需要的中间变量（初始化为0）
    feature_M = np.zeros((row*2, col))  # intermediate variable for computing feature_std2
    # 记录 各个类簇每个特征的方差，row: clusters, col:features
    feature_std2 = np.zeros((row*2, col))  # record variances, being used togather with feature_mean

    # 用于暂存前一个均值，从而方便feature_mean和feature_std2（均值和方差）的在线更新
    old_mean = np.zeros((1, col))  # facilitate the online update of feature_mean and feature_std2

    # 显著性权重s（也叫显著性得分s）
    # 即normalized_salience_weight：每次都是在循环中当场计算，不在此处定义

    # J为cluster的个数
    J = 0

    # 记录每个cluster的内部点的数量，1行row列（最多row个cluster），如果cluster个数少于row，后面再进行删除多余的
    # 每当有一个cluster内多了一个数据点，就 +1
    # !!! 迭代后总量会超过样本数，但是此L仅用于更新均值方差，真正的cluster内部数量可以在performance_cal中计算
    L = np.zeros((1, row*2), dtype=np.int32)

    # Assign记录样本点的分配，1行row列
    # 每列记录每个样本被分配的cluster的index
    Assign = np.zeros((1, row), dtype=np.int64)

    # 警戒参数矩阵，1行row列，用于判断样本点是否满足cluster
    # 可能有无意义的列
    rho_0 = rho * np.ones((1, row*2))

    # 第一轮的第一个样本输入
    Wv[0, :] = M[0, :]
    feature_salience_count[0, np.where(M[0, :] > 0)] += 1  # 凡是特征大于0，则频数 0+1 = 1
    feature_mean[0, :] = M[0, :]  # 由于是第一个样本，则特征均值即为权重  # 注意，此时该类簇每个特征的方差都为0无需更新
    J = J + 1
    L[0, J - 1] = 1
    Assign[0, 0] = J - 1  # 存放索引，注意类簇索引从0开始，所以J-1

    # intermediate variables used in clustering - defined early here
    # temp_a 用于暂存当前在处理的样本向量：In
    # temp_b 用于暂存当前在处理的cluster的特征向量：Wj
    # intersec 用于存放temp_a和temp_b的 公共小部分（即In和Wj取小）
    temp_a = np.zeros((1, col))
    temp_b = np.zeros((1, col))
    intersec = np.zeros((1, col))  # get salient features

    # -----------从第二个样本开始，处理之后的样本---------
    for i in range(1, row):
        # if n % 5000 == 0:
        #     print('Processing data sample {}'.format(n))

        T_max = -1  # the maximun choice value
        winner = -1  # index of the winner cluster

        temp_a[0, :] = M[i, :]

        # 存储某样本对于各个cluster的选择函数T,初始设为-2，保证如果不更改，则一定小于T_winner的初始值-1
        T_values = np.zeros((1, row*2)) - 2

        # compute the similarity with all clusters; find the best-matching cluster
        # 对所有现有的cluster循环，寻找最匹配的类簇
        for j in range(0, J):

            temp_b[0, :] = Wv[j, :]

            # 下面依据 显著性权重s，计算 样本In 和 类簇Cj 的 匹配函数M 和 选择函数T
            # 流程是：
            # 1.先对In和Wj取小，找出结果大于0的索引：intersec_index
            # 2.计算显著性权重s，取出其中inersec_index索引的
            # 3.再去根据公式计算样本In 和 类簇Cj 的 匹配函数M 和 选择函数T
            # 总的来说就是：
            # In和Wj取小，选出大于0的（即后面的inersec_index），再用这些大于0的去和对应的显著性权重相乘，再计算一范数，再计算M与T

            # 1.先对In和Wj取小，找出结果大于0的索引：intersec_index.（后面会用于点乘对应的 显著性权重）
            intersec[0, :] = np.minimum(temp_a, temp_b)
            intersec_index = np.where(intersec[0, :] > 0)

            # 2.对于intersec_index代表的这些特征，计算显著性权重s
            # 由公式可知，需要先算出 频率 和 e的负标准差次方 ，前者衡量特征活跃度，后者衡量特征稳定性。二者加权求和得到显著性权重s
            salience_weight_presence = feature_salience_count[j, :] / L[0, j]
            salience_weight_std = np.exp(-np.sqrt(feature_std2[j, :]))
            # 对他们归一化
            normalized_salience_weight_presence = salience_weight_presence / np.sum(salience_weight_presence)
            normalized_salience_weight_std = salience_weight_std / np.sum(salience_weight_std)
            # 取出其中的intersec_index代表的那些特征
            normalized_salience_weight_presence_intersec = normalized_salience_weight_presence[intersec_index]
            normalized_salience_weight_std_intersec = normalized_salience_weight_std[intersec_index]
            # 计算显著性权重s：lam用于平衡 频率 和 标准差
            normalized_salience_weight = lam * normalized_salience_weight_presence_intersec + \
                                         (1 - lam) * normalized_salience_weight_std_intersec

            # 3.计算M与T
            # 计算分子
            temp = np.sum(intersec[0, intersec_index] * normalized_salience_weight)
            # 计算匹配函数M
            Mj_V = temp / np.sum(temp_a[0, intersec_index] * normalized_salience_weight)
            # 计算选择函数T
            T_values[0, j] = temp / (alpha + np.sum(temp_b[0, intersec_index] * normalized_salience_weight))

            if Mj_V >= rho_0[0, j] and T_values[0, j] >= T_max:
                T_max = T_values[0, j]
                winner = j

        # # AMR策略
        # a = np.where(T_values[0, :] >= T_max)  # 返回一个tuple，a[0]是个数组，里面每个元素是索引 / -2的作用也在此体现出来
        # b = a[0]
        # # 如果有获胜者
        # if winner > -1:
        #     # 对获胜者的rho进行增加
        #     rho_0[0, winner] = (1 + sigma) * rho_0[0, winner]
        #
        #     b = np.delete(b, np.where(b == winner)[0])  # 去除获胜者
        # # 去除获胜者之后（或者没有获胜者），对剩下的这些cluster的rho进行减少
        # rho_0[0, b] = (1 - sigma) * rho_0[0, b]

        # -----------------------------------------------------------------------------------------------
        # Cluster assignment process
        if winner == -1:
            # indicates no cluster passes the vigilance parameter - the rho
            # create a new cluster
            J = J + 1
            Wv[J - 1, :] = M[i, :]
            feature_salience_count[J - 1, np.where(M[i, :] > 0)] += 1
            feature_mean[J - 1, :] = M[i, :]  # 新类簇方差为0，无需更新；只更新均值
            L[0, J - 1] = 1
            Assign[0, i] = J - 1
        else:
            # if winner is found, update cluster weights and do cluster assignment
            # 更新阶段：（不需要更新显著性权重s；因为s在聚类阶段会重新计算；因此只需更新对应的均值mean和方差std2等其他参数）
            # 1.更新获胜者winner类簇 的权重向量
            #   分为：
            #   1.1 用当前winner类簇的 均值 与 方差，按公式来计算自适应学习率
            #   1.2 根据自适应学习率，更新winner类簇 的权重向量
            # 2.在线更新获胜者winner类簇 的 频数、均值、方差

            # 1.更新获胜者winner的cluster权重向量
            # 1.1 用当前winner类簇的 均值(feature_mean) 与 方差(feature_std2)，按公式来计算自适应学习率learning_rate_theta

            # 分别找出 当前方差为0的特征、当前方差不为0的特征
            zero_std_index = np.where(feature_std2[winner, :] == 0)
            non_zero_std_index = np.where(feature_std2[winner, :] > 0)
            # 对于当前方差为0的特征，根据(mean -3*std)>0,(mean +3*std)<1 推测出方差的范围-> std<= min( (mean+0.01)/3, (1-mean)/3 )（具体推导见论文）
            temp1 = (feature_mean[winner, :][zero_std_index] + 0.01) / 3
            temp2 = (1 - (feature_mean[winner, :][zero_std_index]-0.01) ) / 3
            std_for_zero_var = np.minimum(temp1, temp2)

            learning_rate_theta = np.zeros((1, col))
            # 根据公式，计算分为 方差为0的特征/方差不为0的特征 两部分
            # 方差为0的特征
            temp1 = np.square(M[i, :][zero_std_index] - feature_mean[winner, :][zero_std_index])
            temp2 = 2 * (std_for_zero_var ** 2)
            learning_rate_theta[0, zero_std_index] = np.exp(-(temp1 / temp2))
            # 方差不为0的特征
            temp1 = np.square(M[i, :][non_zero_std_index] - feature_mean[winner, :][non_zero_std_index])
            temp2 = 2 * feature_std2[winner, :][non_zero_std_index]
            learning_rate_theta[0, non_zero_std_index] = np.exp(-(temp1 / temp2))

            # 1.2根据自适应学习率，按照公式，更新winner类簇 的权重向量
            vj = np.minimum(M[i, :], feature_mean[winner, :])
            Wv[winner, :] = vj * learning_rate_theta[0, :] + Wv[winner, :] * (1 - learning_rate_theta[0, :])

            # 2.在线更新获胜者winner类簇 的 频数count、均值mean、方差std2
            # 更新频数
            feature_salience_count[winner, np.where(M[i, :] > 0)] += 1
            # 暂存旧的均值mean，后面用
            old_mean[0, :] = feature_mean[winner, :]
            # 在线更新均值mean
            feature_mean[winner, :] = (old_mean[0, :] * L[0, winner] + M[i, :]) / (L[0, winner] + 1)
            # 在线更新方差std2
            feature_M[winner, :] = feature_M[winner, :] + (M[i, :] - old_mean[0, :]) * (
                    M[i, :] - feature_mean[winner, :])
            feature_std2[winner, :] = feature_M[winner, :] / L[0, winner]

            # 类簇信息更新
            Assign[0, i] = winner
            L[0, winner] += 1

    # 评估
    temp_result_dic = Performance_Cal.performance_cal(Assign[0, :], label, J, row, Wv, M)
    # 更新轮数，并存储评价指标
    performance_dic[now_repeat_num] = temp_result_dic

    # ---------------------------------------------------------------------------------------------------
    # 进行多轮SA-ART循环
    # flag记录是否提前终止循环，为True则继续循环
    stop_flag = True

    # 代表最大判断轮数
    # 因为 类簇稳定性判断 之后必须要还有1次迭代，因此 最大判断轮数 是 最大循环次数-1
    # 比如最大循环4次，则只能在第2、3次之后进行判断，第1次不能是因为要先生成类簇结构才能判断得失，第4次不用再判断（因为是最后一次类簇分配）
    Max_Judge_Number = max_repeat_num - 1

    while True:
        stop_flag = False
        now_repeat_num += 1

        # 记录类簇获得/丢失样本情况
        cluster_get = np.zeros(J)

        # print("第{}轮算法开始-----------------".format(now_repeat_num))

        # intermediate variables used in clustering - defined early here
        # temp_a 用于暂存当前在处理的样本向量：In
        # temp_b 用于暂存当前在处理的cluster的特征向量：Wj
        # intersec 用于存放temp_a和temp_b的 公共小部分（即In和Wj取小）
        temp_a = np.zeros((1, col))
        temp_b = np.zeros((1, col))
        intersec = np.zeros((1, col))  # get salient features

        # 开始新一轮SA-ART
        for i in range(0, row):

            # if i % 200 == 0:
            #     print("第{}轮正在处理第{}个样本".format(now_repeat_num, i))

            temp_a[0, :] = M[i, :]
            T_max = -1  # the maximum choice value
            winner = -1  # index of the winner cluster

            pre_Assign = Assign[0, i]  # 该样本上一轮所属于的类簇的编号，-1代表未被分类

            # 存储某样本对于各个cluster的选择函数T,初始设为-2，保证如果不更改，则一定小于T_winner的初始值-1
            T_values = np.zeros((1, row * 2)) - 2

            # 对所有现有的cluster循环，寻找最匹配的类簇
            for j in range(0, J):

                temp_b[0, :] = Wv[j, :]

                # 下面依据 显著性权重s，计算 样本In 和 类簇Cj 的 匹配函数M 和 选择函数T
                # 流程是：
                # 1.先对In和Wj取小，找出结果大于0的索引：intersec_index
                # 2.计算显著性权重s，取出其中inersec_index索引的
                # 3.再去根据公式计算样本In 和 类簇Cj 的 匹配函数M 和 选择函数T
                # 总的来说就是：
                # In和Wj取小，选出大于0的（即后面的inersec_index），再用这些大于0的去和对应的显著性权重相乘，再计算一范数，再计算M与T

                # 1.先对In和Wj取小，找出结果大于0的索引：intersec_index.（后面会用于点乘对应的 显著性权重）
                intersec[0, :] = np.minimum(temp_a, temp_b)
                intersec_index = np.where(intersec[0, :] > 0)

                # 2.对于intersec_index代表的这些特征，计算显著性权重s
                # 由公式可知，需要先算出 频率 和 e的负标准差次方 ，前者衡量特征活跃度，后者衡量特征稳定性。二者加权求和得到显著性权重s
                salience_weight_presence = feature_salience_count[j, :] / L[0, j]
                salience_weight_std = np.exp(-np.sqrt(feature_std2[j, :]))
                # 对他们归一化
                normalized_salience_weight_presence = salience_weight_presence / np.sum(salience_weight_presence)
                normalized_salience_weight_std = salience_weight_std / np.sum(salience_weight_std)
                # 取出其中的intersec_index代表的那些特征
                normalized_salience_weight_presence_intersec = normalized_salience_weight_presence[intersec_index]
                normalized_salience_weight_std_intersec = normalized_salience_weight_std[intersec_index]
                # 计算显著性权重s：lam用于平衡 频率 和 标准差
                normalized_salience_weight = lam * normalized_salience_weight_presence_intersec + \
                                             (1 - lam) * normalized_salience_weight_std_intersec

                # 3.计算M与T
                # 计算分子
                temp = np.sum(intersec[0, intersec_index] * normalized_salience_weight)
                # 计算匹配函数M
                Mj_V = temp / np.sum(temp_a[0, intersec_index] * normalized_salience_weight)
                # 计算选择函数T
                T_values[0, j] = temp / (alpha + np.sum(temp_b[0, intersec_index] * normalized_salience_weight))

                if Mj_V >= rho_0[0, j] and T_values[0, j] >= T_max:
                    T_max = T_values[0, j]
                    winner = j

            # # AMR策略
            # a = np.where(T_values[0, :] >= T_max)  # 返回一个tuple，a[0]是个数组，里面每个元素是索引 / -2的作用也在此体现出来
            # b = a[0]
            # # 如果有获胜者
            # if winner > -1:
            #     # 对获胜者的rho进行增加
            #     rho_0[0, winner] = (1 + sigma) * rho_0[0, winner]
            #
            #     b = np.delete(b, np.where(b == winner)[0])  # 去除获胜者
            # # 去除获胜者之后（或者没有获胜者），对剩下的这些cluster的rho进行减少
            # rho_0[0, b] = (1 - sigma) * rho_0[0, b]

            # -----------------------------------------------------------------------------------------------
            # Cluster assignment process
            if winner == -1:  # 没有cluster超过警戒参数

                # 先统计类簇得/失
                # 因为产生新类簇，所以更新cluster_get,在最后加一个0，用于统计新类簇
                cluster_get = np.concatenate((cluster_get, np.zeros(1)), axis=0)
                # 得到样本的新类簇，值+1
                cluster_get[-1] += 1

                # 如果该样本以前属于某个类簇，则该样本原本的类簇肯定丢失了该样本，故记-1
                if pre_Assign != -1:
                    cluster_get[pre_Assign] -= 1

                # 创建新cluster,更新聚类信息
                J = J + 1
                Wv[J - 1, :] = M[i, :]
                feature_salience_count[J - 1, np.where(M[i, :] > 0)] += 1
                feature_mean[J - 1, :] = M[i, :]  # 新类簇标准差为0，无需更新；只更新均值
                Assign[0, i] = J - 1
                L[0, J - 1] = 1
                stop_flag = True  # 只要类簇assign发生变化则True，继续循环

            else:  # 如果有winner,进行cluster分配并且更新cluster权重参数
                # winner选择策略：不设置（随机选择）

                # 只要有winner，都得更新wv和Assign和L，还有统计值，所以在此处直接先更新即可
                L[0, winner] += 1
                Assign[0, i] = winner

                # if winner is found, update cluster weights and do cluster assignment
                # 更新阶段：（不需要更新显著性权重s；因为s在聚类阶段会重新计算；因此只需更新对应的均值mean和方差std2等其他参数）
                # 1.更新获胜者winner类簇 的权重向量
                #   分为：
                #   1.1 用当前winner类簇的 均值 与 方差，按公式来计算自适应学习率
                #   1.2 根据自适应学习率，更新winner类簇 的权重向量
                # 2.在线更新获胜者winner类簇 的 频数、均值、方差

                # 1.更新获胜者winner的cluster权重向量
                # 1.1 用当前winner类簇的 均值(feature_mean) 与 方差(feature_std2)，按公式来计算自适应学习率learning_rate_theta

                # 分别找出 当前方差为0的特征、当前方差不为0的特征
                zero_std_index = np.where(feature_std2[winner, :] == 0)
                non_zero_std_index = np.where(feature_std2[winner, :] > 0)
                # 对于当前方差为0的特征，根据(mean -3*std)>0,(mean +3*std)<1 推测出方差的范围-> std<= min( (mean+0.01)/3, (1-mean)/3 )（具体推导见论文）
                temp1 = (feature_mean[winner, :][zero_std_index] + 0.01) / 3
                temp2 = (1 - (feature_mean[winner, :][zero_std_index] - 0.01)) / 3
                std_for_zero_var = np.minimum(temp1, temp2)

                learning_rate_theta = np.zeros((1, col))
                # 根据公式，计算分为 方差为0的特征/方差不为0的特征 两部分
                # 方差为0的特征
                temp1 = np.square(M[i, :][zero_std_index] - feature_mean[winner, :][zero_std_index])
                temp2 = 2 * (std_for_zero_var ** 2)
                learning_rate_theta[0, zero_std_index] = np.exp(-(temp1 / temp2))
                # 方差不为0的特征
                temp1 = np.square(M[i, :][non_zero_std_index] - feature_mean[winner, :][non_zero_std_index])
                temp2 = 2 * feature_std2[winner, :][non_zero_std_index]
                learning_rate_theta[0, non_zero_std_index] = np.exp(-(temp1 / temp2))

                # 1.2根据自适应学习率，按照公式，更新winner类簇 的权重向量
                vj = np.minimum(M[i, :], feature_mean[winner, :])
                Wv[winner, :] = vj * learning_rate_theta[0, :] + Wv[winner, :] * (1 - learning_rate_theta[0, :])

                # 2.在线更新获胜者winner类簇 的 频数count、均值mean、方差std2
                # 更新频数
                feature_salience_count[winner, np.where(M[i, :] > 0)] += 1
                # 暂存旧的均值mean，后面用
                old_mean[0, :] = feature_mean[winner, :]
                # 在线更新均值mean
                feature_mean[winner, :] = (old_mean[0, :] * L[0, winner] + M[i, :]) / (L[0, winner] + 1)
                # 在线更新方差std2
                feature_M[winner, :] = feature_M[winner, :] + (M[i, :] - old_mean[0, :]) * (
                        M[i, :] - feature_mean[winner, :])
                feature_std2[winner, :] = feature_M[winner, :] / L[0, winner]

                # 下面的判断都是用来处理cluster_get 和 stop_flag的  ！！！！！！
                # 如果该样本上一轮属于某个类簇
                if pre_Assign != -1:
                    # 判断新分配的类簇，是否还是之前的类簇
                    if pre_Assign != winner:
                        # 如果之前的分配和现在的分配不一样，则之前类簇丢失样本，winner类簇获得样本
                        cluster_get[pre_Assign] -= 1
                        cluster_get[winner] += 1
                        # print("第{}个样本在第{}轮更新了所属cluster".format(i, now_repeat_num))
                        stop_flag = True  # 只要类簇assign发生变化则True，继续循环

                    else:
                        # 如果之前的分配和现在的分配一样，则该类簇没获得新的，不+也不-
                        # 类簇Assign没更新，stop_flag也不需要更新
                        # cluster权重也已经更新了，则没啥事要做
                        pass

                else:  # 该样本上一轮不属于任何类簇
                    cluster_get[winner] += 1
                    # print("第{}个未分配样本在第{}轮归属了cluster".format(i, now_repeat_num))
                    stop_flag = True  # 因为该样本上一轮不属于任何类簇，本轮属于了该类簇，所有assign发生变化，则True

        # 每次完成SA-ART执行后，此时每个样本都有所属类簇，可以计算此时的聚类指标
        # 此时所有样本都有所属类簇，可以计算可靠的指标。但是要注意，此时可能存在“消亡类簇”，需要在聚类指标计算方法中去进行判断，并且
        temp_result_dic = Performance_Cal.performance_cal(Assign[0, :], label, J, row, Wv, M)
        # 更新轮数，并存储评价指标
        performance_dic[now_repeat_num] = temp_result_dic

        # 判断是否结束循环：结束条件是 达到最大迭代次数 或者 本轮的类簇分配情况没有改变
        if (now_repeat_num == max_repeat_num) or (not stop_flag):
            # 达到了终止条件，则上一次FuzzyART即为最终结果，下面开始收尾，不在迭代
            # 不再进行不稳定类簇的统计和删除
            # 最后处理一下“类簇消亡现象” （目的是更新权重矩阵等参数，去掉消亡的类簇，确保参数的正确性，以及画图的可靠性！！！！！）
            # print('达到最大判断轮次:{}'.format(Max_Judge_Number))
            # 原本的类簇集合（J个，编号0~J-1）
            old_cluster_set = set(range(0, J))
            # 现在的类簇集合(用set去重)
            new_assign = set(Assign[0, :])
            # 相减得到 消亡的类簇 的编号
            difference = old_cluster_set - new_assign
            # 如果集合不为空，则说明出现了类簇消亡的情况
            if difference != set():
                # print('出现了类簇消亡的情况,消亡的类簇编号为 {}'.format(difference))

                # 新的类簇数目
                new_J = len(new_assign)

                # 新的Assign
                # 从0到new_J的列表
                new_cluster_list = list(range(0, new_J))
                # 构建转换字典
                replace_dic = dict(zip(new_assign, new_cluster_list))
                # 对于所有样本，将老label(Assign)替换成新label
                temp_Assign = Assign[0, :].copy()
                for index, value in enumerate(temp_Assign):
                    temp_Assign[index] = replace_dic[value]

                # 新的Wv
                new_Wv_index = list(new_assign)
                new_Wv = np.zeros((row, col))
                new_Wv[0:new_J, :] = Wv[new_Wv_index, :]

                # 新的rho_0，注意此处不再调整rho，只是因为删除了消亡的类簇，需要把相应的rho进行对应
                new_rho_index = list(new_assign)
                new_rho_0 = rho * np.ones((1, row))
                new_rho_0[0, 0:new_J] = rho_0[0, new_rho_index]

                # 将新的J、Wv、Assign更新到循环变量上
                J = new_J
                Assign[0, :] = temp_Assign
                Wv = new_Wv
                rho_0 = new_rho_0

                # ！！！ 此时现在的J、Assign、Wv、rho_0便是最终达到最大迭代次数后，最终的聚类结果。可以作为方法的最终输出，输出出去。

                # 更新cluster_want_to_show
                if cluster_want_to_show in replace_dic.keys():
                    cluster_want_to_show = replace_dic[cluster_want_to_show]
                elif cluster_want_to_show == -1:
                    continue
                else:
                    cluster_want_to_show = -2

            # 可以返回了。需要画图就按照“类簇消亡”更新后的Assign，Wv和rho_0画图就好
            # print("第{}轮聚类算法结束-------------".format(now_repeat_num))
            # FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M,
            #                                          assign=Assign[0, :],
            #                                          cluster_weight=Wv[0:J, :],
            #                                          cluster_rho=rho_0[0, 0:J],
            #                                          cluster_want_to_show=cluster_want_to_show)

            return performance_dic

        # 如果没达到终止条件，则不会结束循环，下面还得继续 类簇稳定性判断
        else:
            # 进行 类簇稳定性判断
            # -----------------------------------类簇稳定性判断-----------------------------------
            # print('进行第{}轮迭代中的稳定性判断步骤'.format(now_repeat_num))

            # 出现的“类簇消亡现象”(迭代后cluster中样本数为0)，可知其得失数必然<0,一定会被认为是不稳定类簇，因此不需要单独考虑“类簇消亡”
            # 按照类簇得失情况，把 <0 的认为是不稳定类簇，获得其index
            unstable_cluster = np.where(cluster_get < 0)[0]
            unstable_cluster_number = len(unstable_cluster)
            # 如果不稳定类簇数目>0，即 存在不稳定类簇    注意如果不存在不稳定类簇，则不进入该if，直接回去继续循环
            if unstable_cluster_number > 0:
                # print('存在不稳定类簇:{}'.format(unstable_cluster))

                # 原本的类簇集合（J个，编号0~J-1）
                old_cluster_set = set(range(0, J))
                # 不稳定类簇集合
                unstable_cluster_set = set(unstable_cluster)
                # 要保留的类簇集合(用set去重)
                new_assign = old_cluster_set - unstable_cluster_set

                # 新的类簇数目
                new_J = len(new_assign)

                # 新的Assign，其中-1代表不属于任何类簇（下一轮会被重新聚类获得新类簇label）
                # 此处必须要更新，因为如果Assign不更新，则可能里面存在已经消亡的类簇的索引
                # 然而 下一轮判断聚类分配是否发生变化时（即flag），需要用到正确的，对应好的Assign：被聚类的样本则存放对应类簇的索引，未被聚类的则为-1
                # 从0到new_J的列表
                new_cluster_list = list(range(0, new_J))
                # 构建转换字典
                replace_dic = dict(zip(new_assign, new_cluster_list))
                # 对于所有样本，将老label(Assign)替换成新label
                temp_Assign = Assign[0, :].copy()
                for index, value in enumerate(temp_Assign):
                    if temp_Assign[index] in unstable_cluster_set:
                        # 如果所属类簇在不稳定类簇中，则置为-1，表示该样本不属于任何类簇，未聚类
                        # 设为-1，这样在下面调用画图方法时，如果cluster_want_to_show 为-1，则会将这些聚类样本特殊显示
                        temp_Assign[index] = -1
                    else:
                        # 如果所属类簇是稳定的，则按照字典更新 新的label
                        temp_Assign[index] = replace_dic[value]

                # 新的Wv
                new_Wv_index = list(new_assign)
                new_Wv = np.zeros((row*2, col))
                new_Wv[0:new_J, :] = Wv[new_Wv_index, :]

                # 得到稳定类簇后，创建新的rho_0
                new_rho_0 = rho * np.ones((1, row*2))
                for key, value in replace_dic.items():
                    # 稳定类簇之前的rho
                    pre_rho = rho_0[0, key]
                    # rho调整策略：利用参数expand_ratio适当调小稳定类簇的rho
                    new_rho = pre_rho * (1 - expand_ratio)
                    # 赋值
                    new_rho_0[0, value] = new_rho

                # 将新的J、Wv、Assign更新到循环变量上
                J = new_J
                Assign[0, :] = temp_Assign
                Wv = new_Wv
                rho_0 = new_rho_0

                # 更新cluster_want_to_show
                if cluster_want_to_show in replace_dic.keys():
                    cluster_want_to_show = replace_dic[cluster_want_to_show]
                elif cluster_want_to_show == -1:
                    pass
                else:
                    cluster_want_to_show = -2

                # 更新SA-ART的统计信息，把不稳定类簇的统计信息删除，留下稳定的，并更新index
                # 包括 频数、均值、方差、L

                # 1.频数
                new_feature_salience_count_index = list(new_assign)
                new_feature_salience_count = np.zeros((row*2, col))
                new_feature_salience_count[0:new_J, :] = feature_salience_count[new_feature_salience_count_index, :]

                # 2.均值
                new_feature_mean_index = list(new_assign)
                new_feature_mean = np.zeros((row * 2, col))
                new_feature_mean[0:new_J, :] = feature_mean[new_feature_mean_index, :]

                # 3.方差
                # 计算 方差 过程中需要的中间变量（初始化为0）
                new_feature_M_index = list(new_assign)
                new_feature_M = np.zeros((row * 2, col))
                new_feature_M[0:new_J, :] = feature_M[new_feature_M_index, :]
                # 记录 各个类簇每个特征的方差，row: clusters, col:features
                new_feature_std2_index = list(new_assign)
                new_feature_std2 = np.zeros((row * 2, col))
                new_feature_std2[0:new_J, :] = feature_std2[new_feature_std2_index, :]

                # 4.L
                # 得到稳定类簇后，创建新的L
                new_L = np.zeros((1, row * 2), dtype=np.int32)
                for key, value in replace_dic.items():
                    # 稳定类簇之前的L值
                    pre_L_num = L[0, key]
                    # 赋值
                    L[0,value] = pre_L_num

                # print("不稳定类簇删除完毕")

                # print("第{}轮聚类算法结束-------------".format(now_repeat_num))
                # FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M,
                #                                          assign=Assign[0, :],
                #                                          cluster_weight=Wv[0:J, :],
                #                                          cluster_rho=rho_0[0, 0:J],
                #                                          cluster_want_to_show=cluster_want_to_show)

    # print("算法全部执行完毕")
    # # 循环算法执行完毕，绘图
    # FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M,
    #                                          assign=Assign[0, :],
    #                                          cluster_weight=Wv[0:J, :],
    #                                          cluster_rho=rho_0[0, 0:J],
    #                                          cluster_want_to_show=cluster_want_to_show)

    return performance_dic


def si_art_repeat_quick_stop_without_draw_with_AMR(M, label, rho, lam, sigma, alpha, expand_ratio, max_repeat_num, cluster_want_to_show=-1):
    """
    @param M: numpy arrary; m*n 特征矩阵; m是实例个数 ，n是特征数
    @param label: 维度：m，代表样本所属真实类别（从0开始）
    @param rho: 警戒参数(0-1)
    @param sigma: the percentage to enlarge or shrink vigilance region(用于AMR中控制扩大或缩小警戒区域的程度)
    @param delta: 很小的数值，用于CM-ART策略中微微收缩边界
    @param beta: # has no significant impact on performance with a moderate value of [0.4,0.7]
    @param alpha: 避免除以0
    @return:
    """

    NAME = 'si_art'
    # print(NAME + "算法开始------------------")
    # -----------------------------------------------------------------------------------------------------------------------
    # Input parameters
    # no need to tune; used in choice function;
    # to avoid too small cluster weights (resulted by the learning method of ART; should be addressed sometime);
    # give priority to choosing denser clusters
    alpha = alpha

    # rho needs carefully tune; used to shape the inter-cluster similarity;
    # rho_v = 0.7 indicates an object will not be clustered to a cluster with visual similarity lower than 0.7
    # rho = 0.6

    # -----------------------------------------------------------------------------------------------------------------------
    # Initialization

    # complement coding
    M = np.concatenate([M, 1 - M], 1)

    # get data sizes
    row, col = M.shape

    # 接收每轮的结果情况
    performance_dic = {}
    # 保表示当前重复执行的轮数
    now_repeat_num = 0

    # -----------------------------------------------------------------------------------------------------------------------
    # Clustering process

    # print(NAME + "第一轮算法开始---------------------")
    now_repeat_num += 1

    # 用第一个样本初始化第一个cluster
    # Wv存放cluster权重参数，row行col列
    # 如果后面cluster小于row个，则把多余的行去掉
    Wv = np.zeros((row*2, col))

    # 记录各个类簇的每个特征的 频数（不是频率），row: clusters, col:features
    # 认为一个特征>0则该特征频数+1
    feature_salience_count = np.zeros((row*2, col))

    # 记录各个类簇的每个特征的 均值，row: clusters, col:features
    feature_mean = np.zeros((row*2, col))

    # 计算 方差 过程中需要的中间变量（初始化为0）
    feature_M = np.zeros((row*2, col))  # intermediate variable for computing feature_std2
    # 记录 各个类簇每个特征的方差，row: clusters, col:features
    feature_std2 = np.zeros((row*2, col))  # record variances, being used togather with feature_mean

    # 用于暂存前一个均值，从而方便feature_mean和feature_std2（均值和方差）的在线更新
    old_mean = np.zeros((1, col))  # facilitate the online update of feature_mean and feature_std2

    # 显著性权重s（也叫显著性得分s）
    # 即normalized_salience_weight：每次都是在循环中当场计算，不在此处定义

    # J为cluster的个数
    J = 0

    # 记录每个cluster的内部点的数量，1行row列（最多row个cluster），如果cluster个数少于row，后面再进行删除多余的
    # 每当有一个cluster内多了一个数据点，就 +1
    # !!! 迭代后总量会超过样本数，但是此L仅用于更新均值方差，真正的cluster内部数量可以在performance_cal中计算
    L = np.zeros((1, row*2), dtype=np.int32)

    # Assign记录样本点的分配，1行row列
    # 每列记录每个样本被分配的cluster的index
    Assign = np.zeros((1, row), dtype=np.int64)

    # 警戒参数矩阵，1行row列，用于判断样本点是否满足cluster
    # 可能有无意义的列
    rho_0 = rho * np.ones((1, row*2))

    # 第一轮的第一个样本输入
    Wv[0, :] = M[0, :]
    feature_salience_count[0, np.where(M[0, :] > 0)] += 1  # 凡是特征大于0，则频数 0+1 = 1
    feature_mean[0, :] = M[0, :]  # 由于是第一个样本，则特征均值即为权重  # 注意，此时该类簇每个特征的方差都为0无需更新
    J = J + 1
    L[0, J - 1] = 1
    Assign[0, 0] = J - 1  # 存放索引，注意类簇索引从0开始，所以J-1

    # intermediate variables used in clustering - defined early here
    # temp_a 用于暂存当前在处理的样本向量：In
    # temp_b 用于暂存当前在处理的cluster的特征向量：Wj
    # intersec 用于存放temp_a和temp_b的 公共小部分（即In和Wj取小）
    temp_a = np.zeros((1, col))
    temp_b = np.zeros((1, col))
    intersec = np.zeros((1, col))  # get salient features

    # -----------从第二个样本开始，处理之后的样本---------
    for i in range(1, row):
        # if n % 5000 == 0:
        #     print('Processing data sample {}'.format(n))

        T_max = -1  # the maximun choice value
        winner = -1  # index of the winner cluster

        temp_a[0, :] = M[i, :]

        # 存储某样本对于各个cluster的选择函数T,初始设为-2，保证如果不更改，则一定小于T_winner的初始值-1
        T_values = np.zeros((1, row*2)) - 2

        # compute the similarity with all clusters; find the best-matching cluster
        # 对所有现有的cluster循环，寻找最匹配的类簇
        for j in range(0, J):

            temp_b[0, :] = Wv[j, :]

            # 下面依据 显著性权重s，计算 样本In 和 类簇Cj 的 匹配函数M 和 选择函数T
            # 流程是：
            # 1.先对In和Wj取小，找出结果大于0的索引：intersec_index
            # 2.计算显著性权重s，取出其中inersec_index索引的
            # 3.再去根据公式计算样本In 和 类簇Cj 的 匹配函数M 和 选择函数T
            # 总的来说就是：
            # In和Wj取小，选出大于0的（即后面的inersec_index），再用这些大于0的去和对应的显著性权重相乘，再计算一范数，再计算M与T

            # 1.先对In和Wj取小，找出结果大于0的索引：intersec_index.（后面会用于点乘对应的 显著性权重）
            intersec[0, :] = np.minimum(temp_a, temp_b)
            intersec_index = np.where(intersec[0, :] > 0)

            # 2.对于intersec_index代表的这些特征，计算显著性权重s
            # 由公式可知，需要先算出 频率 和 e的负标准差次方 ，前者衡量特征活跃度，后者衡量特征稳定性。二者加权求和得到显著性权重s
            salience_weight_presence = feature_salience_count[j, :] / L[0, j]
            salience_weight_std = np.exp(-np.sqrt(feature_std2[j, :]))
            # 对他们归一化
            normalized_salience_weight_presence = salience_weight_presence / np.sum(salience_weight_presence)
            normalized_salience_weight_std = salience_weight_std / np.sum(salience_weight_std)
            # 取出其中的intersec_index代表的那些特征
            normalized_salience_weight_presence_intersec = normalized_salience_weight_presence[intersec_index]
            normalized_salience_weight_std_intersec = normalized_salience_weight_std[intersec_index]
            # 计算显著性权重s：lam用于平衡 频率 和 标准差
            normalized_salience_weight = lam * normalized_salience_weight_presence_intersec + \
                                         (1 - lam) * normalized_salience_weight_std_intersec

            # 3.计算M与T
            # 计算分子
            temp = np.sum(intersec[0, intersec_index] * normalized_salience_weight)
            # 计算匹配函数M
            Mj_V = temp / np.sum(temp_a[0, intersec_index] * normalized_salience_weight)
            # 计算选择函数T
            T_values[0, j] = temp / (alpha + np.sum(temp_b[0, intersec_index] * normalized_salience_weight))

            if Mj_V >= rho_0[0, j] and T_values[0, j] >= T_max:
                T_max = T_values[0, j]
                winner = j

        # AMR策略
        a = np.where(T_values[0, :] >= T_max)  # 返回一个tuple，a[0]是个数组，里面每个元素是索引 / -2的作用也在此体现出来
        b = a[0]
        # 如果有获胜者
        if winner > -1:
            # 对获胜者的rho进行增加
            rho_0[0, winner] = (1 + sigma) * rho_0[0, winner]

            b = np.delete(b, np.where(b == winner)[0])  # 去除获胜者
        # 去除获胜者之后（或者没有获胜者），对剩下的这些cluster的rho进行减少
        rho_0[0, b] = (1 - sigma) * rho_0[0, b]

        # -----------------------------------------------------------------------------------------------
        # Cluster assignment process
        if winner == -1:
            # indicates no cluster passes the vigilance parameter - the rho
            # create a new cluster
            J = J + 1
            Wv[J - 1, :] = M[i, :]
            feature_salience_count[J - 1, np.where(M[i, :] > 0)] += 1
            feature_mean[J - 1, :] = M[i, :]  # 新类簇方差为0，无需更新；只更新均值
            L[0, J - 1] = 1
            Assign[0, i] = J - 1
        else:
            # if winner is found, update cluster weights and do cluster assignment
            # 更新阶段：（不需要更新显著性权重s；因为s在聚类阶段会重新计算；因此只需更新对应的均值mean和方差std2等其他参数）
            # 1.更新获胜者winner类簇 的权重向量
            #   分为：
            #   1.1 用当前winner类簇的 均值 与 方差，按公式来计算自适应学习率
            #   1.2 根据自适应学习率，更新winner类簇 的权重向量
            # 2.在线更新获胜者winner类簇 的 频数、均值、方差

            # 1.更新获胜者winner的cluster权重向量
            # 1.1 用当前winner类簇的 均值(feature_mean) 与 方差(feature_std2)，按公式来计算自适应学习率learning_rate_theta

            # 分别找出 当前方差为0的特征、当前方差不为0的特征
            zero_std_index = np.where(feature_std2[winner, :] == 0)
            non_zero_std_index = np.where(feature_std2[winner, :] > 0)
            # 对于当前方差为0的特征，根据(mean -3*std)>0,(mean +3*std)<1 推测出方差的范围-> std<= min( (mean+0.01)/3, (1-mean)/3 )（具体推导见论文）
            temp1 = (feature_mean[winner, :][zero_std_index] + 0.01) / 3
            temp2 = (1 - (feature_mean[winner, :][zero_std_index]-0.01) ) / 3
            std_for_zero_var = np.minimum(temp1, temp2)

            learning_rate_theta = np.zeros((1, col))
            # 根据公式，计算分为 方差为0的特征/方差不为0的特征 两部分
            # 方差为0的特征
            temp1 = np.square(M[i, :][zero_std_index] - feature_mean[winner, :][zero_std_index])
            temp2 = 2 * (std_for_zero_var ** 2)
            learning_rate_theta[0, zero_std_index] = np.exp(-(temp1 / temp2))
            # 方差不为0的特征
            temp1 = np.square(M[i, :][non_zero_std_index] - feature_mean[winner, :][non_zero_std_index])
            temp2 = 2 * feature_std2[winner, :][non_zero_std_index]
            learning_rate_theta[0, non_zero_std_index] = np.exp(-(temp1 / temp2))

            # 1.2根据自适应学习率，按照公式，更新winner类簇 的权重向量
            vj = np.minimum(M[i, :], feature_mean[winner, :])
            Wv[winner, :] = vj * learning_rate_theta[0, :] + Wv[winner, :] * (1 - learning_rate_theta[0, :])

            # 2.在线更新获胜者winner类簇 的 频数count、均值mean、方差std2
            # 更新频数
            feature_salience_count[winner, np.where(M[i, :] > 0)] += 1
            # 暂存旧的均值mean，后面用
            old_mean[0, :] = feature_mean[winner, :]
            # 在线更新均值mean
            feature_mean[winner, :] = (old_mean[0, :] * L[0, winner] + M[i, :]) / (L[0, winner] + 1)
            # 在线更新方差std2
            feature_M[winner, :] = feature_M[winner, :] + (M[i, :] - old_mean[0, :]) * (
                    M[i, :] - feature_mean[winner, :])
            feature_std2[winner, :] = feature_M[winner, :] / L[0, winner]

            # 类簇信息更新
            Assign[0, i] = winner
            L[0, winner] += 1

    # 评估
    temp_result_dic = Performance_Cal.performance_cal(Assign[0, :], label, J, row, Wv, M)
    # 更新轮数，并存储评价指标
    performance_dic[now_repeat_num] = temp_result_dic

    # ---------------------------------------------------------------------------------------------------
    # 进行多轮SA-ART循环
    # flag记录是否提前终止循环，为True则继续循环
    stop_flag = True

    # 代表最大判断轮数
    # 因为 类簇稳定性判断 之后必须要还有1次迭代，因此 最大判断轮数 是 最大循环次数-1
    # 比如最大循环4次，则只能在第2、3次之后进行判断，第1次不能是因为要先生成类簇结构才能判断得失，第4次不用再判断（因为是最后一次类簇分配）
    Max_Judge_Number = max_repeat_num - 1

    while True:
        stop_flag = False
        now_repeat_num += 1

        # 记录类簇获得/丢失样本情况
        cluster_get = np.zeros(J)

        # print("第{}轮算法开始-----------------".format(now_repeat_num))

        # intermediate variables used in clustering - defined early here
        # temp_a 用于暂存当前在处理的样本向量：In
        # temp_b 用于暂存当前在处理的cluster的特征向量：Wj
        # intersec 用于存放temp_a和temp_b的 公共小部分（即In和Wj取小）
        temp_a = np.zeros((1, col))
        temp_b = np.zeros((1, col))
        intersec = np.zeros((1, col))  # get salient features

        # 开始新一轮SA-ART
        for i in range(0, row):

            # if i % 200 == 0:
            #     print("第{}轮正在处理第{}个样本".format(now_repeat_num, i))

            temp_a[0, :] = M[i, :]
            T_max = -1  # the maximum choice value
            winner = -1  # index of the winner cluster

            pre_Assign = Assign[0, i]  # 该样本上一轮所属于的类簇的编号，-1代表未被分类

            # 存储某样本对于各个cluster的选择函数T,初始设为-2，保证如果不更改，则一定小于T_winner的初始值-1
            T_values = np.zeros((1, row * 2)) - 2

            # 对所有现有的cluster循环，寻找最匹配的类簇
            for j in range(0, J):

                temp_b[0, :] = Wv[j, :]

                # 下面依据 显著性权重s，计算 样本In 和 类簇Cj 的 匹配函数M 和 选择函数T
                # 流程是：
                # 1.先对In和Wj取小，找出结果大于0的索引：intersec_index
                # 2.计算显著性权重s，取出其中inersec_index索引的
                # 3.再去根据公式计算样本In 和 类簇Cj 的 匹配函数M 和 选择函数T
                # 总的来说就是：
                # In和Wj取小，选出大于0的（即后面的inersec_index），再用这些大于0的去和对应的显著性权重相乘，再计算一范数，再计算M与T

                # 1.先对In和Wj取小，找出结果大于0的索引：intersec_index.（后面会用于点乘对应的 显著性权重）
                intersec[0, :] = np.minimum(temp_a, temp_b)
                intersec_index = np.where(intersec[0, :] > 0)

                # 2.对于intersec_index代表的这些特征，计算显著性权重s
                # 由公式可知，需要先算出 频率 和 e的负标准差次方 ，前者衡量特征活跃度，后者衡量特征稳定性。二者加权求和得到显著性权重s
                salience_weight_presence = feature_salience_count[j, :] / L[0, j]
                salience_weight_std = np.exp(-np.sqrt(feature_std2[j, :]))
                # 对他们归一化
                normalized_salience_weight_presence = salience_weight_presence / np.sum(salience_weight_presence)
                normalized_salience_weight_std = salience_weight_std / np.sum(salience_weight_std)
                # 取出其中的intersec_index代表的那些特征
                normalized_salience_weight_presence_intersec = normalized_salience_weight_presence[intersec_index]
                normalized_salience_weight_std_intersec = normalized_salience_weight_std[intersec_index]
                # 计算显著性权重s：lam用于平衡 频率 和 标准差
                normalized_salience_weight = lam * normalized_salience_weight_presence_intersec + \
                                             (1 - lam) * normalized_salience_weight_std_intersec

                # 3.计算M与T
                # 计算分子
                temp = np.sum(intersec[0, intersec_index] * normalized_salience_weight)
                # 计算匹配函数M
                Mj_V = temp / np.sum(temp_a[0, intersec_index] * normalized_salience_weight)
                # 计算选择函数T
                T_values[0, j] = temp / (alpha + np.sum(temp_b[0, intersec_index] * normalized_salience_weight))

                if Mj_V >= rho_0[0, j] and T_values[0, j] >= T_max:
                    T_max = T_values[0, j]
                    winner = j

            # AMR策略
            a = np.where(T_values[0, :] >= T_max)  # 返回一个tuple，a[0]是个数组，里面每个元素是索引 / -2的作用也在此体现出来
            b = a[0]
            # 如果有获胜者
            if winner > -1:
                # 对获胜者的rho进行增加
                rho_0[0, winner] = (1 + sigma) * rho_0[0, winner]

                b = np.delete(b, np.where(b == winner)[0])  # 去除获胜者
            # 去除获胜者之后（或者没有获胜者），对剩下的这些cluster的rho进行减少
            rho_0[0, b] = (1 - sigma) * rho_0[0, b]

            # -----------------------------------------------------------------------------------------------
            # Cluster assignment process
            if winner == -1:  # 没有cluster超过警戒参数

                # 先统计类簇得/失
                # 因为产生新类簇，所以更新cluster_get,在最后加一个0，用于统计新类簇
                cluster_get = np.concatenate((cluster_get, np.zeros(1)), axis=0)
                # 得到样本的新类簇，值+1
                cluster_get[-1] += 1

                # 如果该样本以前属于某个类簇，则该样本原本的类簇肯定丢失了该样本，故记-1
                if pre_Assign != -1:
                    cluster_get[pre_Assign] -= 1

                # 创建新cluster,更新聚类信息
                J = J + 1
                Wv[J - 1, :] = M[i, :]
                feature_salience_count[J - 1, np.where(M[i, :] > 0)] += 1
                feature_mean[J - 1, :] = M[i, :]  # 新类簇标准差为0，无需更新；只更新均值
                Assign[0, i] = J - 1
                L[0, J - 1] = 1
                stop_flag = True  # 只要类簇assign发生变化则True，继续循环

            else:  # 如果有winner,进行cluster分配并且更新cluster权重参数
                # winner选择策略：不设置（随机选择）

                # 只要有winner，都得更新wv和Assign和L，还有统计值，所以在此处直接先更新即可
                L[0, winner] += 1
                Assign[0, i] = winner

                # if winner is found, update cluster weights and do cluster assignment
                # 更新阶段：（不需要更新显著性权重s；因为s在聚类阶段会重新计算；因此只需更新对应的均值mean和方差std2等其他参数）
                # 1.更新获胜者winner类簇 的权重向量
                #   分为：
                #   1.1 用当前winner类簇的 均值 与 方差，按公式来计算自适应学习率
                #   1.2 根据自适应学习率，更新winner类簇 的权重向量
                # 2.在线更新获胜者winner类簇 的 频数、均值、方差

                # 1.更新获胜者winner的cluster权重向量
                # 1.1 用当前winner类簇的 均值(feature_mean) 与 方差(feature_std2)，按公式来计算自适应学习率learning_rate_theta

                # 分别找出 当前方差为0的特征、当前方差不为0的特征
                zero_std_index = np.where(feature_std2[winner, :] == 0)
                non_zero_std_index = np.where(feature_std2[winner, :] > 0)
                # 对于当前方差为0的特征，根据(mean -3*std)>0,(mean +3*std)<1 推测出方差的范围-> std<= min( (mean+0.01)/3, (1-mean)/3 )（具体推导见论文）
                temp1 = (feature_mean[winner, :][zero_std_index] + 0.01) / 3
                temp2 = (1 - (feature_mean[winner, :][zero_std_index] - 0.01)) / 3
                std_for_zero_var = np.minimum(temp1, temp2)

                learning_rate_theta = np.zeros((1, col))
                # 根据公式，计算分为 方差为0的特征/方差不为0的特征 两部分
                # 方差为0的特征
                temp1 = np.square(M[i, :][zero_std_index] - feature_mean[winner, :][zero_std_index])
                temp2 = 2 * (std_for_zero_var ** 2)
                learning_rate_theta[0, zero_std_index] = np.exp(-(temp1 / temp2))
                # 方差不为0的特征
                temp1 = np.square(M[i, :][non_zero_std_index] - feature_mean[winner, :][non_zero_std_index])
                temp2 = 2 * feature_std2[winner, :][non_zero_std_index]
                learning_rate_theta[0, non_zero_std_index] = np.exp(-(temp1 / temp2))

                # 1.2根据自适应学习率，按照公式，更新winner类簇 的权重向量
                vj = np.minimum(M[i, :], feature_mean[winner, :])
                Wv[winner, :] = vj * learning_rate_theta[0, :] + Wv[winner, :] * (1 - learning_rate_theta[0, :])

                # 2.在线更新获胜者winner类簇 的 频数count、均值mean、方差std2
                # 更新频数
                feature_salience_count[winner, np.where(M[i, :] > 0)] += 1
                # 暂存旧的均值mean，后面用
                old_mean[0, :] = feature_mean[winner, :]
                # 在线更新均值mean
                feature_mean[winner, :] = (old_mean[0, :] * L[0, winner] + M[i, :]) / (L[0, winner] + 1)
                # 在线更新方差std2
                feature_M[winner, :] = feature_M[winner, :] + (M[i, :] - old_mean[0, :]) * (
                        M[i, :] - feature_mean[winner, :])
                feature_std2[winner, :] = feature_M[winner, :] / L[0, winner]

                # 下面的判断都是用来处理cluster_get 和 stop_flag的  ！！！！！！
                # 如果该样本上一轮属于某个类簇
                if pre_Assign != -1:
                    # 判断新分配的类簇，是否还是之前的类簇
                    if pre_Assign != winner:
                        # 如果之前的分配和现在的分配不一样，则之前类簇丢失样本，winner类簇获得样本
                        cluster_get[pre_Assign] -= 1
                        cluster_get[winner] += 1
                        # print("第{}个样本在第{}轮更新了所属cluster".format(i, now_repeat_num))
                        stop_flag = True  # 只要类簇assign发生变化则True，继续循环

                    else:
                        # 如果之前的分配和现在的分配一样，则该类簇没获得新的，不+也不-
                        # 类簇Assign没更新，stop_flag也不需要更新
                        # cluster权重也已经更新了，则没啥事要做
                        pass

                else:  # 该样本上一轮不属于任何类簇
                    cluster_get[winner] += 1
                    # print("第{}个未分配样本在第{}轮归属了cluster".format(i, now_repeat_num))
                    stop_flag = True  # 因为该样本上一轮不属于任何类簇，本轮属于了该类簇，所有assign发生变化，则True

        # 每次完成SA-ART执行后，此时每个样本都有所属类簇，可以计算此时的聚类指标
        # 此时所有样本都有所属类簇，可以计算可靠的指标。但是要注意，此时可能存在“消亡类簇”，需要在聚类指标计算方法中去进行判断，并且
        temp_result_dic = Performance_Cal.performance_cal(Assign[0, :], label, J, row, Wv, M)
        # 更新轮数，并存储评价指标
        performance_dic[now_repeat_num] = temp_result_dic

        # 判断是否结束循环：结束条件是 达到最大迭代次数 或者 本轮的类簇分配情况没有改变
        if (now_repeat_num == max_repeat_num) or (not stop_flag):
            # 达到了终止条件，则上一次FuzzyART即为最终结果，下面开始收尾，不在迭代
            # 不再进行不稳定类簇的统计和删除
            # 最后处理一下“类簇消亡现象” （目的是更新权重矩阵等参数，去掉消亡的类簇，确保参数的正确性，以及画图的可靠性！！！！！）
            # print('达到最大判断轮次:{}'.format(Max_Judge_Number))
            # 原本的类簇集合（J个，编号0~J-1）
            old_cluster_set = set(range(0, J))
            # 现在的类簇集合(用set去重)
            new_assign = set(Assign[0, :])
            # 相减得到 消亡的类簇 的编号
            difference = old_cluster_set - new_assign
            # 如果集合不为空，则说明出现了类簇消亡的情况
            if difference != set():
                # print('出现了类簇消亡的情况,消亡的类簇编号为 {}'.format(difference))

                # 新的类簇数目
                new_J = len(new_assign)

                # 新的Assign
                # 从0到new_J的列表
                new_cluster_list = list(range(0, new_J))
                # 构建转换字典
                replace_dic = dict(zip(new_assign, new_cluster_list))
                # 对于所有样本，将老label(Assign)替换成新label
                temp_Assign = Assign[0, :].copy()
                for index, value in enumerate(temp_Assign):
                    temp_Assign[index] = replace_dic[value]

                # 新的Wv
                new_Wv_index = list(new_assign)
                new_Wv = np.zeros((row, col))
                new_Wv[0:new_J, :] = Wv[new_Wv_index, :]

                # 新的rho_0，注意此处不再调整rho，只是因为删除了消亡的类簇，需要把相应的rho进行对应
                new_rho_index = list(new_assign)
                new_rho_0 = rho * np.ones((1, row))
                new_rho_0[0, 0:new_J] = rho_0[0, new_rho_index]

                # 将新的J、Wv、Assign更新到循环变量上
                J = new_J
                Assign[0, :] = temp_Assign
                Wv = new_Wv
                rho_0 = new_rho_0

                # ！！！ 此时现在的J、Assign、Wv、rho_0便是最终达到最大迭代次数后，最终的聚类结果。可以作为方法的最终输出，输出出去。

                # 更新cluster_want_to_show
                if cluster_want_to_show in replace_dic.keys():
                    cluster_want_to_show = replace_dic[cluster_want_to_show]
                elif cluster_want_to_show == -1:
                    continue
                else:
                    cluster_want_to_show = -2

            # 可以返回了。需要画图就按照“类簇消亡”更新后的Assign，Wv和rho_0画图就好
            # print("第{}轮聚类算法结束-------------".format(now_repeat_num))
            # FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M,
            #                                          assign=Assign[0, :],
            #                                          cluster_weight=Wv[0:J, :],
            #                                          cluster_rho=rho_0[0, 0:J],
            #                                          cluster_want_to_show=cluster_want_to_show)

            return performance_dic

        # 如果没达到终止条件，则不会结束循环，下面还得继续 类簇稳定性判断
        else:
            # 进行 类簇稳定性判断
            # -----------------------------------类簇稳定性判断-----------------------------------
            # print('进行第{}轮迭代中的稳定性判断步骤'.format(now_repeat_num))

            # 出现的“类簇消亡现象”(迭代后cluster中样本数为0)，可知其得失数必然<0,一定会被认为是不稳定类簇，因此不需要单独考虑“类簇消亡”
            # 按照类簇得失情况，把 <0 的认为是不稳定类簇，获得其index
            unstable_cluster = np.where(cluster_get < 0)[0]
            unstable_cluster_number = len(unstable_cluster)
            # 如果不稳定类簇数目>0，即 存在不稳定类簇    注意如果不存在不稳定类簇，则不进入该if，直接回去继续循环
            if unstable_cluster_number > 0:
                # print('存在不稳定类簇:{}'.format(unstable_cluster))

                # 原本的类簇集合（J个，编号0~J-1）
                old_cluster_set = set(range(0, J))
                # 不稳定类簇集合
                unstable_cluster_set = set(unstable_cluster)
                # 要保留的类簇集合(用set去重)
                new_assign = old_cluster_set - unstable_cluster_set

                # 新的类簇数目
                new_J = len(new_assign)

                # 新的Assign，其中-1代表不属于任何类簇（下一轮会被重新聚类获得新类簇label）
                # 此处必须要更新，因为如果Assign不更新，则可能里面存在已经消亡的类簇的索引
                # 然而 下一轮判断聚类分配是否发生变化时（即flag），需要用到正确的，对应好的Assign：被聚类的样本则存放对应类簇的索引，未被聚类的则为-1
                # 从0到new_J的列表
                new_cluster_list = list(range(0, new_J))
                # 构建转换字典
                replace_dic = dict(zip(new_assign, new_cluster_list))
                # 对于所有样本，将老label(Assign)替换成新label
                temp_Assign = Assign[0, :].copy()
                for index, value in enumerate(temp_Assign):
                    if temp_Assign[index] in unstable_cluster_set:
                        # 如果所属类簇在不稳定类簇中，则置为-1，表示该样本不属于任何类簇，未聚类
                        # 设为-1，这样在下面调用画图方法时，如果cluster_want_to_show 为-1，则会将这些聚类样本特殊显示
                        temp_Assign[index] = -1
                    else:
                        # 如果所属类簇是稳定的，则按照字典更新 新的label
                        temp_Assign[index] = replace_dic[value]

                # 新的Wv
                new_Wv_index = list(new_assign)
                new_Wv = np.zeros((row*2, col))
                new_Wv[0:new_J, :] = Wv[new_Wv_index, :]

                # 得到稳定类簇后，创建新的rho_0
                new_rho_0 = rho * np.ones((1, row*2))
                for key, value in replace_dic.items():
                    # 稳定类簇之前的rho
                    pre_rho = rho_0[0, key]
                    # rho调整策略：利用参数expand_ratio适当调小稳定类簇的rho
                    new_rho = pre_rho * (1 - expand_ratio)
                    # 赋值
                    new_rho_0[0, value] = new_rho

                # 将新的J、Wv、Assign更新到循环变量上
                J = new_J
                Assign[0, :] = temp_Assign
                Wv = new_Wv
                rho_0 = new_rho_0

                # 更新cluster_want_to_show
                if cluster_want_to_show in replace_dic.keys():
                    cluster_want_to_show = replace_dic[cluster_want_to_show]
                elif cluster_want_to_show == -1:
                    pass
                else:
                    cluster_want_to_show = -2

                # 更新SA-ART的统计信息，把不稳定类簇的统计信息删除，留下稳定的，并更新index
                # 包括 频数、均值、方差、L

                # 1.频数
                new_feature_salience_count_index = list(new_assign)
                new_feature_salience_count = np.zeros((row*2, col))
                new_feature_salience_count[0:new_J, :] = feature_salience_count[new_feature_salience_count_index, :]

                # 2.均值
                new_feature_mean_index = list(new_assign)
                new_feature_mean = np.zeros((row * 2, col))
                new_feature_mean[0:new_J, :] = feature_mean[new_feature_mean_index, :]

                # 3.方差
                # 计算 方差 过程中需要的中间变量（初始化为0）
                new_feature_M_index = list(new_assign)
                new_feature_M = np.zeros((row * 2, col))
                new_feature_M[0:new_J, :] = feature_M[new_feature_M_index, :]
                # 记录 各个类簇每个特征的方差，row: clusters, col:features
                new_feature_std2_index = list(new_assign)
                new_feature_std2 = np.zeros((row * 2, col))
                new_feature_std2[0:new_J, :] = feature_std2[new_feature_std2_index, :]

                # 4.L
                # 得到稳定类簇后，创建新的L
                new_L = np.zeros((1, row * 2), dtype=np.int32)
                for key, value in replace_dic.items():
                    # 稳定类簇之前的L值
                    pre_L_num = L[0, key]
                    # 赋值
                    L[0,value] = pre_L_num

                # print("不稳定类簇删除完毕")

                # print("第{}轮聚类算法结束-------------".format(now_repeat_num))
                # FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M,
                #                                          assign=Assign[0, :],
                #                                          cluster_weight=Wv[0:J, :],
                #                                          cluster_rho=rho_0[0, 0:J],
                #                                          cluster_want_to_show=cluster_want_to_show)

    # print("算法全部执行完毕")
    # # 循环算法执行完毕，绘图
    # FuzzyART_Plot.plot_2D_draw_weight_and_VR(data_matrix=M,
    #                                          assign=Assign[0, :],
    #                                          cluster_weight=Wv[0:J, :],
    #                                          cluster_rho=rho_0[0, 0:J],
    #                                          cluster_want_to_show=cluster_want_to_show)

    return performance_dic





# def iterative_sa_art(M,label,rho):
#     '''
#     % M: numpy arrary; m*n feature matrix; m is number of objects and n is number of visual features
#     %rho: the vigilance parameter
#     %save_path_root: path to save clustering results for further analysis
#     '''
#     performance_dic = {}
#     lam = 0.9
#     NAME = 'iterative_sa_art'
#     max_iteration = 50
#
#     #get data sizes
#     row, col = M.shape
#
#
# # -----------------------------------------------------------------------------------------------------------------------
# # Clustering process
#
#     print(NAME + "algorithm starts")
#
#     #create initial cluster with the first data sample
#         #initialize cluster parameters
#     Wv = np.zeros((row, col))
#     feature_salience_count = np.zeros((row, col))  # count the frequency that a feature has been selected as salient ones
#     feature_mean=np.zeros((row, col)) #record means of each features in terms of clusters - row: clusters, col:features
#     feature_M = np.zeros((row, col)) #intermediate variable for computing feature_std2
#     feature_std2 =np.zeros((row, col)) #record variances, being used togather with feature_mean
#     old_mean = np.zeros((1,col)) #facilitate the online update of feature_mean and feature_std2
#     salience_weight_prob = np.zeros((1, col))
#     J = 0  # number of clusters
#     L = np.zeros((1,row))  # size of clusters; note we set to the maximun number of cluster, i.e. number of rows
#     Assign = np.zeros((1,row), dtype=np.int)  # the cluster assignment of objects
#         #first cluster
#     print('Iteration 1: Processing data sample 1')
#     Wv[0, :] = M[0, :]
#     feature_salience_count[0,np.where(M[0, :]>0)] += 1
#     feature_mean[0,:] = M[0, :]
#     J = 1
#     L[0,J-1] = 1
#     Assign[0,0] = J-1 #note that python array index trickily starts from 0
#
#         #intermediate variables used in clustering - defined early here
#     intersec = np.zeros((1, col)) # get salient features
#     temp_a = np.zeros((1, col))
#     temp_b = np.zeros((1, col))
#
#     #processing other objects
#     for n in range(1,row):
#
#         print('Iteration 1: Processing data sample %d' % (n+1))
#
#         T_max = -1 #the maximun choice value
#         winner = -1 #index of the winner cluster
#
#         temp_a[0, :] = M[n, :]
#
#         #compute the similarity with all clusters; find the best-matching cluster
#         for j in range(0,J):
#
#             temp_b[0, :] = Wv[j, :]
#
#             #compute the match value according to salient features of clusters
#             intersec[0,:] = np.minimum(temp_a,temp_b) #get histogram intersection between input and cluster
#             intersec_index = np.where(intersec[0,:] > 0) #use common features as salient features for similarity measure
#             salience_index = np.where(feature_salience_count[j,:]>0) # using cluster j's salient features for similarity measure - not stable in early stage
#
#                 #different measures for importance of features
#                     # count the presence as an active feature; more weights give to active features
#             salience_weight_presence = feature_salience_count[j, :] / L[0, j]
#             salience_weight_presence_intersec = salience_weight_presence[intersec_index]
#                     # count the stability of feature value as representatives; more weights give to stable features
#             salience_weight_std = np.exp(-np.sqrt(feature_std2[j, :]))
#             salience_weight_std_intersec = salience_weight_std[intersec_index]
#
#                 #weight fusion
#             normalized_salience_weight = (salience_weight_presence_intersec /np.sum(salience_weight_presence[salience_index])) *lam + (salience_weight_std_intersec / np.sum(salience_weight_std[salience_index]))*(1-lam)
#
#
#                 #compute match value
#             Mj_V = np.sum((intersec[0, intersec_index] / temp_a[0, intersec_index]) * normalized_salience_weight)
#
#             if Mj_V - rho >= 0:
#                 Tj = np.sum((intersec[0, intersec_index] / temp_b[0, intersec_index]) * normalized_salience_weight)
#                 if Tj - T_max >= 0:
#                     T_max = Tj
#                     winner = j
#
#         #Cluster assignment process
#         if winner == -1: #indicates no cluster passes the vigilance parameter - the rho
#             #create a new cluster
#             J = J + 1
#             Wv[J - 1, :] = M[n, :]
#             feature_salience_count[J-1, np.where(M[n, :] > 0)] += 1
#             feature_mean[J-1, :] = M[n, :]
#             L[0, J - 1] = 1
#             Assign[0,n] = J - 1
#         else: #if winner is found, do cluster assignment and update cluster weights
#             # update cluster weights
#                 #compute likelihood of input features being members of winner - it determines how much we learn from the input for individual features
#             zero_std_index = np.where(feature_std2[winner,:] == 0)
#             non_zero_std_index = np.where(feature_std2[winner, :] > 0)
#             std_for_zero_var = np.minimum((feature_mean[winner, :][zero_std_index]+0.01)/3,(1-feature_mean[winner, :][zero_std_index])/3) #it is based on the functions that (mean -3*std)>0,(mean +3*std)<1
#
#             salience_weight_prob[0, zero_std_index] = np.exp(
#                 -(np.square(M[n, :][zero_std_index] - feature_mean[winner, :][zero_std_index]) / (2 * (std_for_zero_var**2))))
#             salience_weight_prob[0, non_zero_std_index] = np.exp(-(np.square(
#                 M[n, :][non_zero_std_index] - feature_mean[winner, :][non_zero_std_index]) / (2 * feature_std2[winner, :][non_zero_std_index])))
#
#             Wv[winner, :] = np.minimum(M[n, :],feature_mean[winner,:]) * salience_weight_prob[0,:] + Wv[winner, :] * (1-salience_weight_prob[0,:])
#
#
#             # update of statistics and cluster assignment
#                 # update salient features - count shared features only
#             feature_min_values = np.minimum(M[n, :], Wv[winner, :])
#             salient_index = np.where(feature_min_values > 0)
#             feature_salience_count[winner, salient_index] += 1
#                 # compute new mean&variance for each feature of winner
#             old_mean[0, :] = feature_mean[winner, :]
#             feature_mean[winner, :] = (old_mean[0, :] * L[0, winner] + M[n, :]) / (L[0, winner] + 1)
#             feature_M[winner, :] = feature_M[winner, :] + (M[n, :] - old_mean[0, :]) * (M[n, :] - feature_mean[winner, :])
#             feature_std2[winner, :] = feature_M[winner, :] / L[0, winner]
#                 # cluster assignment
#             L[0, winner] += 1
#             Assign[0, n] = winner
#
#     # 评估
#     temp_result_dic = Performance_Cal.performance_cal(Assign[0, :], label, J, row, Wv, M)
#     # 更新轮数，并存储评价指标
#     performance_dic[0] = temp_result_dic
#
#
#     for iteration in range(1, max_iteration):
#         for n in range(0, row):
#             print('Iteration ' + str(iteration+1)+': Processing data sample %d' % (n+1))
#
#             T_max = -1 #the maximun choice value
#             winner = -1 #index of the winner cluster
#
#             temp_a[0, :] = M[n, :]
#
#             #compute the similarity with all clusters; find the best-matching cluster
#             for j in range(0,J):
#
#                 temp_b[0, :] = Wv[j, :]
#
#                 #compute the match value according to salient features of clusters
#                 intersec[0,:] = np.minimum(temp_a,temp_b) #get histogram intersection between input and cluster
#                 intersec_index = np.where(intersec[0,:] > 0) #use common features as salient features for similarity measure
#                 salience_index = np.where(feature_salience_count[j,:]>0) # using cluster j's salient features for similarity measure - not stable in early stage
#
#                     #different measures for importance of features
#                         # count the presence as an active feature; more weights give to active features
#                 salience_weight_presence = feature_salience_count[j, :] / L[0, j]
#                 salience_weight_presence_intersec = salience_weight_presence[intersec_index]
#                         # count the stability of feature value as representatives; more weights give to stable features
#                 salience_weight_std = np.exp(-np.sqrt(feature_std2[j, :]))
#                 salience_weight_std_intersec = salience_weight_std[intersec_index]
#
#
#                     #weight fusion
#                 normalized_salience_weight = (salience_weight_presence_intersec /np.sum(salience_weight_presence[salience_index])+ salience_weight_std_intersec / np.sum(salience_weight_std[salience_index])) / 2
#
#
#                     #compute match value
#                 Mj_V = np.sum((intersec[0, intersec_index] / temp_a[0, intersec_index]) * normalized_salience_weight)
#
#                 if Mj_V - rho >= 0:
#                     Tj = np.sum((intersec[0, intersec_index] / temp_b[0, intersec_index]) * normalized_salience_weight)
#                     if Tj - T_max >= 0:
#                         T_max = Tj
#                         winner = j
#
#             #Cluster assignment process
#             if winner == -1: #indicates no cluster passes the vigilance parameter - the rho
#                 #create a new cluster
#                 J = J + 1
#                 Wv[J - 1, :] = M[n, :]
#                 feature_salience_count[J-1, np.where(M[n, :] > 0)] += 1
#                 feature_mean[J-1, :] = M[n, :]
#                 L[0, J - 1] = 1
#                 Assign[0,n] = J - 1
#             else: #if winner is found, do cluster assignment and update cluster weights
#                 # update cluster weights
#                     #compute likelihood of input features being members of winner - it determines how much we learn from the input for individual features
#                 zero_std_index = np.where(feature_std2[winner,:] == 0)
#                 non_zero_std_index = np.where(feature_std2[winner, :] > 0)
#                 std_for_zero_var = np.minimum((feature_mean[winner, :][zero_std_index]+0.01)/3,(1-feature_mean[winner, :][zero_std_index])/3) #it is based on the functions that (mean -3*std)>0,(mean +3*std)<1
#
#                 salience_weight_prob[0, zero_std_index] = np.exp(
#                     -(np.square(M[n, :][zero_std_index] - feature_mean[winner, :][zero_std_index]) / (2 * (std_for_zero_var**2))))
#                 salience_weight_prob[0, non_zero_std_index] = np.exp(-(np.square(
#                     M[n, :][non_zero_std_index] - feature_mean[winner, :][non_zero_std_index]) / (2 * feature_std2[winner, :][non_zero_std_index])))
#
#                 Wv[winner, :] = np.minimum(M[n, :],feature_mean[winner,:]) * salience_weight_prob[0,:] + Wv[winner, :] * (1-salience_weight_prob[0,:])
#
#                 # update of statistics and cluster assignment
#                     # update salient features - count shared features only
#                 feature_min_values = np.minimum(M[n, :], Wv[winner, :])
#                 salient_index = np.where(feature_min_values > 0)
#                 feature_salience_count[winner, salient_index] += 1
#                     # compute new mean&variance for each feature of winner
#                 old_mean[0, :] = feature_mean[winner, :]
#                 feature_mean[winner, :] = (old_mean[0, :] * L[0, winner] + M[n, :]) / (L[0, winner] + 1)
#                 feature_M[winner, :] = feature_M[winner, :] + (M[n, :] - old_mean[0, :]) * (M[n, :] - feature_mean[winner, :])
#                 feature_std2[winner, :] = feature_M[winner, :] / L[0, winner]
#                     # cluster assignment
#                 L[0, winner] += 1
#                 Assign[0, n] = winner
#
#         # save results
#         # 评估
#         temp_result_dic = Performance_Cal.performance_cal(Assign[0, :], label, J, row, Wv, M)
#         # 更新轮数，并存储评价指标
#         performance_dic[iteration] = temp_result_dic
#
#     print("algorithm ends")
#
#
#     return performance_dic




if __name__ == '__main__':
    data_feature, data_label = DataLoad.load_data_Synthetic_control(shuffle=True,seed=20)
    performance = si_art_repeat_quick_stop_without_draw_without_AMR(data_feature,
                                                                 data_label,
                                                                 rho=0.91,
                                                                 lam=0.5,
                                                                 alpha=0.001,
                                                                 expand_ratio=0.1,
                                                                 max_repeat_num=10,
                                                                 cluster_want_to_show=-2)
    # performance = si_art_repeat_quick_stop_without_draw_with_AMR(data_feature,
    #                                                              data_label,
    #                                                              rho=0.8,
    #                                                              lam=0.5,
    #                                                              sigma=0.0001,
    #                                                              alpha=0.001,
    #                                                              expand_ratio=0.1,
    #                                                              max_repeat_num=10,
    #                                                              cluster_want_to_show=-2)
    FuzzyART_Plot.draw_performance_line_chart_with_iteration(performance)


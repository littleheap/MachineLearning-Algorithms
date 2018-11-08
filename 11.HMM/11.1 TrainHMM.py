import math

infinite = float(-2 ** 31)


def log_normalize(a):
    s = 0
    for x in a:
        s += x
    if s == 0:
        print("Error..from log_normalize.")
        return
    s = math.log(s)
    for i in range(len(a)):
        if a[i] == 0:
            a[i] = infinite
        else:
            a[i] = math.log(a[i]) - s


def log_sum(a):
    if not a:  # a为空
        return infinite
    m = max(a)
    s = 0
    for t in a:
        s += math.exp(t - m)
    return m + math.log(s)


def calc_alpha(pi, A, B, o, alpha):
    for i in range(4):
        alpha[0][i] = pi[i] + B[i][ord(o[0])]
    T = len(o)
    temp = [0 for i in range(4)]
    del i
    for t in range(1, T):
        for i in range(4):
            for j in range(4):
                temp[j] = (alpha[t - 1][j] + A[j][i])
            alpha[t][i] = log_sum(temp)
            alpha[t][i] += B[i][ord(o[t])]


def calc_beta(pi, A, B, o, beta):
    T = len(o)
    for i in range(4):
        beta[T - 1][i] = 1
    temp = [0 for i in range(4)]
    del i
    for t in range(T - 2, -1, -1):
        for i in range(4):
            beta[t][i] = 0
            for j in range(4):
                temp[j] = A[i][j] + B[j][ord(o[t + 1])] + beta[t + 1][j]
            beta[t][i] += log_sum(temp)


def calc_gamma(alpha, beta, gamma):
    for t in range(len(alpha)):
        for i in range(4):
            gamma[t][i] = alpha[t][i] + beta[t][i]
        s = log_sum(gamma[t])
        for i in range(4):
            gamma[t][i] -= s


def calc_ksi(alpha, beta, A, B, o, ksi):
    T = len(alpha)
    temp = [0 for x in range(16)]
    for t in range(T - 1):
        k = 0
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] = alpha[t][i] + A[i][j] + B[j][ord(o[t + 1])] + beta[t + 1][j]
                temp[k] = ksi[t][i][j]
                k += 1
        s = log_sum(temp)
        for i in range(4):
            for j in range(4):
                ksi[t][i][j] -= s


def bw(pi, A, B, alpha, beta, gamma, ksi, o):
    T = len(alpha)
    for i in range(4):
        pi[i] = gamma[0][i]
    s1 = [0 for x in range(T - 1)]
    s2 = [0 for x in range(T - 1)]
    for i in range(4):
        for j in range(4):
            for t in range(T - 1):
                s1[t] = ksi[t][i][j]
                s2[t] = gamma[t][i]
            A[i][j] = log_sum(s1) - log_sum(s2)
    s1 = [0 for x in range(T)]
    s2 = [0 for x in range(T)]
    for i in range(4):
        print("bw", i)

        for k in range(65536):
            valid = 0
            if k % 10000 == 0:
                print("bw - k", k)

            for t in range(T):
                if ord(o[t]) == k:
                    s1[valid] = gamma[t][i]
                    valid += 1
                s2[t] = gamma[t][i]
            if valid == 0:
                B[i][k] = infinite
            else:
                B[i][k] = log_sum(s1[:valid]) - log_sum(s2)


def baum_welch(pi, A, B):
    f = open(".\\1.txt")
    sentence = f.read()[3:].decode('utf-8')
    f.close()
    T = len(sentence)
    alpha = [[0 for i in range(4)] for t in range(T)]
    beta = [[0 for i in range(4)] for t in range(T)]
    gamma = [[0 for i in range(4)] for t in range(T)]
    ksi = [[[0 for j in range(4)] for i in range(4)] for t in range(T - 1)]
    for time in range(3):
        print("calc_alpha")
        calc_alpha(pi, A, B, sentence, alpha)  # alpha(t,i):给定lamda，在时刻t的状态为i且观测到o(1),o(2)...o(t)的概率
        print("calc_beta")
        calc_beta(pi, A, B, sentence, beta)  # beta(t,i)：给定lamda和时刻t的状态i，观测到o(t+1),o(t+2)...oT的概率
        print("calc_gamma")
        calc_gamma(alpha, beta, gamma)  # gamma(t,i)：给定lamda和O，在时刻t状态位于i的概率
        print("calc_ksi")
        calc_ksi(alpha, beta, A, B, sentence, ksi)  # ksi(t,i,j)：给定lamda和O，在时刻t状态位于i且在时刻i+1，状态位于j的概率
        print("bw")
        bw(pi, A, B, alpha, beta, gamma, ksi, sentence)
        print("time", time)
        print("Pi:", pi)
        print("A", A)


def mle():  # B(Begin) / M(Middle) / E(End) / S(Single)
    pi = [0] * 4  # npi[i]：i状态的个数
    a = [[0] * 4 for x in range(4)]  # na[i][j]：从i状态到j状态的转移个数
    b = [[0] * 65536 for x in range(4)]  # nb[i][o]：从i状态到o字符的个数
    f = open(".\\pku_training.utf8", encoding='utf-8')
    data = f.read()[3:]
    f.close()
    # print(data)
    # 获取每一个词
    tokens = data.split('  ')
    # print(tokens)
    last_q = 2
    iii = 0
    old_progress = 0
    print('进度：')
    for k, token in enumerate(tokens):
        # 打印进度
        progress = float(k) / float(len(tokens))
        if progress > old_progress + 0.1:
            print('%.3f' % progress)
            old_progress = progress
        token = token.strip()
        # 获取当前词长度
        n = len(token)
        if n <= 0:
            continue
        # 长度为1表示为Single，对应下标为3
        if n == 1:
            pi[3] += 1  # pi矩阵第三类+1
            a[last_q][3] += 1  # A矩阵：上一个词的结束(last_q)到当前状态(Single)+1
            b[3][ord(token[0])] += 1  # B矩阵：Single状态到token[0]的字+1
            last_q = 3
            continue
        # 初始向量，长度不是1，一定多一组Begin和End
        pi[0] += 1  # Begin+1
        pi[2] += 1  # End+1
        pi[1] += (n - 2)  # Middle+(n-2)
        # 转移矩阵
        a[last_q][0] += 1  # 上一个状态到Begin+1
        last_q = 2  # 上一个状态设置为End
        #  长度为2，则Begin直接到End
        if n == 2:
            a[0][2] += 1
        else:  # 如果长度大于2
            a[0][1] += 1  # Begin到Middle+1
            a[1][1] += (n - 3)  # Middle到Middle+(n-3)
            a[1][2] += 1  # Middle到End+1
        # 发射矩阵
        b[0][ord(token[0])] += 1  # 从Begin到开始词+1
        b[2][ord(token[n - 1])] += 1  # 从End到最后一个词+1
        for i in range(1, n - 1):  # 中间那些字Middle+1
            b[1][ord(token[i])] += 1
    # 对数正则化
    log_normalize(pi)
    for i in range(4):
        log_normalize(a[i])
        log_normalize(b[i])
    return [pi, a, b]


def list_write(f, v):
    for a in v:
        f.write(str(a))
        f.write(' ')
    f.write('\n')


def save_parameter(pi, A, B):
    f_pi = open(".\\pi.txt", "w")
    list_write(f_pi, pi)
    f_pi.close()
    f_A = open(".\\A.txt", "w")
    for a in A:
        list_write(f_A, a)
    f_A.close()
    f_B = open(".\\B.txt", "w")
    for b in B:
        list_write(f_B, b)
    f_B.close()


if __name__ == "__main__":
    pi, A, B = mle()
    save_parameter(pi, A, B)
    print("训练完成...")

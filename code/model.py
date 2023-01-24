import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torchsnooper

def pad(tensor, length, cuda_flag):
    if isinstance(tensor, Variable):
        if cuda_flag:
            var = tensor.cuda()
        else:
            var = tensor
        if length > var.size(0):
            if cuda_flag:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if cuda_flag:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def feature_transfer(bank_s_, bank_p_, seq_lengths, cuda_flag=False):

    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()
    if cuda_flag:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)
    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)  # cumsum表示前面的元素的累和

    bank_s = torch.stack(
        [pad(bank_s_.narrow(0, s, l), max_len, cuda_flag) for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0
    ).transpose(0, 1)
    bank_p = torch.stack(
        [pad(bank_p_.narrow(0, s, l), max_len, cuda_flag) for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0
    ).transpose(0, 1)

    return bank_s, bank_p

class SpeakerReasonModule(nn.Module):
    def __init__(self, in_channels=200, processing_steps=0, num_layers=1):
        """
        Reasoning Module
        processing_steps
        """
        super(SpeakerReasonModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.fc = nn.Linear(200,200)
        self.fc1 = nn.Linear(400,200)
        if processing_steps > 0:
            self.lstm = nn.LSTM(self.out_channels, self.in_channels, num_layers)  # 400,200,1
            self.lstm.reset_parameters()

    # @torchsnooper.snoop()
    def forward(self,speaker_cos_index,speaker_local_index,bank_s,length,cuda_flag):

        for i in range(len(speaker_local_index)):
            #对于一个长时记忆序列而言 这里先保存当前的长时记忆序列
            A_count_index  =1
            B_count_index = 1
            speaker = []

            flage = False
            for j in range(len(speaker_cos_index[i])):
                if j ==0:
                    speaker.append(speaker_local_index[i][j])
                    flage = True
                    A= speaker_local_index[i][j]
                if torch.mean(speaker_local_index[i][j])!=torch.mean(A) and flage==True:
                    speaker.append(speaker_local_index[i][j])
                if torch.mean(speaker_local_index[i][j])==torch.mean(A):
                    current_index = 0
                    # A_count_index =1+A_count_index
                else:
                    current_index = 1
                    # B_count_index=1+current_index
                long_speaker_m = speaker[current_index]
                if cuda_flag==True:
                    long_speaker_m = long_speaker_m.cuda()
                    temp_speaker_i = speaker_cos_index[i][j].cuda()
                else:
                    long_speaker_m = long_speaker_m
                    temp_speaker_i = speaker_cos_index[i][j]
                # long_speaker_m.backward()
                a =torch.mm(temp_speaker_i,long_speaker_m)
                t = bank_s[j][i].unsqueeze(0).unsqueeze(0)
                q_t = self.fc(t)
                h0 = (a.unsqueeze(0 ),
                     a.new_zeros((self.num_layers, 1, self.in_channels)))

                for xt in range(self.processing_steps):

                    out_put,h0 = self.lstm(q_t,h0)
                    #attention_score
                    temp_1 = out_put.squeeze(0)
                    x_1 = temp_1.mm(long_speaker_m.T)
                    attention_score = torch.softmax(x_1,1)
                    x = attention_score.mm(long_speaker_m)
                    q_t_1 = torch.cat([x,q_t.squeeze(0)],1)

                    out_put_reason = self.fc1(q_t_1)
                    q_t = out_put_reason.unsqueeze(0)
                if current_index==0:
                    updata_index = A_count_index - 1
                    A_count_index=A_count_index+1
                else:
                    updata_index = B_count_index - 1
                    B_count_index = B_count_index + 1
                p_situ_1 = self.fc(out_put_reason)
                temp_p_situ_2 = speaker[current_index][updata_index]
                p_situ_2 = self.fc(temp_p_situ_2)
                # q_situ_2 = self.dropout(q_situ_2)
                z_1 = torch.tanh(p_situ_1 + p_situ_2)
                q_situ_update = z_1 * temp_p_situ_2
                value= torch.cat([speaker[current_index][:updata_index],q_situ_update],0)
                value2 = torch.cat([value,speaker[current_index][updata_index+1:]],0)
                # speaker[current_index] = speaker[current_index].index_put(indices=indices, values=q_situ_update)
                speaker[current_index] = value2
                # speaker[current_index][updata_index]=q_situ_update

                if j == 0:
                    utterence_speaker = out_put_reason
                else:
                    utterence_speaker = torch.cat([utterence_speaker,out_put_reason],0)
            if i==0:
                # if utterence_speaker.size(0) == length:
                #     utterence_tensor_speaker = utterence_speaker.unsqueeze(0)
                # else:
                    utterence_tensor_speaker = pad(utterence_speaker,length,cuda_flag=cuda_flag).unsqueeze(0)
            else:
                # if utterence_speaker.size(0)==length:
                #     temp=utterence_speaker.unsqueeze(0)
                # else:
                temp =  pad(utterence_speaker,length,cuda_flag=cuda_flag).unsqueeze(0)

                utterence_tensor_speaker =  torch.cat([utterence_tensor_speaker,temp],0)


        return utterence_tensor_speaker

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
class ReasonModule(nn.Module):
    def __init__(self, in_channels=200, processing_steps=0, num_layers=1):
        """
        Reasoning Module
        processing_steps：
        """
        super(ReasonModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        if processing_steps > 0:
            self.lstm = nn.LSTM(self.out_channels, self.in_channels, num_layers)  # 400,200,1
            self.lstm.reset_parameters()

    def forward(self, x, batch, q_star,bank_s_list,bank_s,index,cuda):

        if self.processing_steps <= 0: return q_star

        #开始对cosStu的计算
        length = [i.size(1) for i in bank_s_list]
        sum_count = 0
        zero_tensor = torch.zeros(self.in_channels)
        a_sit=None
        for i in range(len(bank_s_list)):

            begin=sum_count
            end = begin+length[i]
            if index<length[i]:
                cos = bank_s_list[i][index].unsqueeze(0)
                H = x[begin:end]
                out_cos = torch.mm(cos,H)
            else:
                out_cos=zero_tensor
            if i == 0:
                a_sit = out_cos
            else:

                if len(a_sit.shape) == 1:
                    a_sit = a_sit.unsqueeze(0)
                if len(out_cos.shape)==1:
                    out_cos = out_cos.unsqueeze(0)
                # print(a_sit.size())
                # print("out_cos",out_cos.size())
                if cuda:
                    a_sit = a_sit.cuda()
                    out_cos = out_cos.cuda()
                a_sit=torch.cat([a_sit,out_cos],0)
            sum_count+=length[i]

        batch_size = batch.max().item() + 1
        a_sit = a_sit.unsqueeze(0)
        # h的初始值
        h = (a_sit,
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        # unsqueeze，对数据维度进行扩充
        for i in range(self.processing_steps):
            # q：[32, 200]
            q, h = self.lstm(q_star.unsqueeze(0), h)  # lstm(h, c)
            q = q.view(batch_size, self.in_channels)
            # attention
            # q[batch]：[1653, 200]
            # x：[1653, 200]
            # q[batch]是每个conversation的第t个utterance重复（当前conversation包含的utterance数量）次
            # x是每个conversation的每个utterance
            # x * q[batch]：表示每个conversation的第t个utterance和当前conversation的所有utterance做dot product
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            # r：[32, 200]
            # q_star：[32, 400]
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)
        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class CognitionNetwork(nn.Module):
    def __init__(self, n_features=200, n_classes=7, dropout=0.2, cuda_flag=False, reason_steps=None):
        """
        Multi-turn Reasoning Modules
        """
        super(CognitionNetwork, self).__init__()
        self.cuda_flag = cuda_flag
        self.q_fc = nn.Linear(n_features, n_features * 2)
        self.m_fc1= nn.Linear(n_features, n_features)
        self.m_fc2 = nn.Linear(n_features, n_features)
        self.steps = reason_steps if reason_steps is not None else [0, 0]
        # 2条通道各自的推理模块
        self.SpeakerReasonModel=SpeakerReasonModule(in_channels=n_features, processing_steps=self.steps[1], num_layers=1)

        self.reason_modules = nn.ModuleList([
            ReasonModule(in_channels=n_features, processing_steps=self.steps[0], num_layers=1),
            ReasonModule(in_channels=n_features, processing_steps=self.steps[1], num_layers=1)
        ])
        #
        self.encode1 = nn.Linear(n_features * 2, n_features)
        self.encode2 = nn.Linear(n_features * 2, n_features)
        self.w1_1 = nn.Linear(n_features*2,n_features)
        self.w1_2 = nn.Linear(n_features,n_features)
        self.w2_1 = nn.Linear(n_features*2,n_features)
        self.w2_2 = nn.Linear(n_features,n_features)

        self.dropout = nn.Dropout(dropout)
        self.smax_fc = nn.Linear(n_features * 3, n_classes)


    def cos_sim_math(self, tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
    def make_cos(self,bank_s_c,seq_lengths):
        bank_s_list = []
        bank_s_c = bank_s_c.transpose(0,1)
        for i in range(len(seq_lengths)):
            start = 0
            end = seq_lengths[i]
            temp_seq = bank_s_c[i][start:end][:]
            all_current_tensor = []
            for j in range((seq_lengths[i])):
                count = 0
                for k in range((seq_lengths[i])):
                    if j == k:
                        pass
                    else:
                        if count == 0:
                            cos = F.cosine_similarity(temp_seq[j], temp_seq[k], dim=0)

                            count =count+ 1

                            cos_tensor = torch.tensor([cos])
                        else:
                            tmep = F.cosine_similarity(temp_seq[j], temp_seq[k], dim=0)

                            tmep_1 = torch.tensor([tmep])
                            cos_tensor = torch.cat([cos_tensor, tmep_1], 0)
                # 开始组合 一个对话的内容
                if j == 0:
                    temp_cos_tensor = torch.softmax(cos_tensor, 0)
                    tem_x = torch.cat([torch.tensor([0.0]), temp_cos_tensor], 0)
                    all_current_tensor.append(tem_x.unsqueeze(0))


                else:
                    temp_cos_tensor = torch.softmax(cos_tensor, 0)
                    begin = temp_cos_tensor[:j]
                    tem_A = torch.cat([begin, torch.tensor([0.0])], 0)
                    tem_C = torch.cat([tem_A, temp_cos_tensor[j:]], 0)
                    x = tem_C.unsqueeze(0)

                    all_current_tensor.append(x)

            bank_s_list.append(all_current_tensor)


        return bank_s_list
    def make_cos_person(self,bank_s_c):
        bank_s_list = []
        # bank_s_c = bank_s_c.transpose(0,1)
        for i in range(bank_s_c.size(0)):
            start = 0
            a = bank_s_c[i]
            for t in range(bank_s_c[i].size(0)):
                hidden = a[t]
                if a[t].sum() == 0:
                    end_index = t
                    break

            temp_seq = a[start:end_index+1][:]
            all_current_tensor = []
            for j in range(end_index):
                count = 0
                for k in range(end_index):
                    if j == 0 and k == 0:
                        pass
                    else:
                        if count == 0:
                            cos = F.cosine_similarity(temp_seq[j], temp_seq[k], dim=0)

                            count =count+ 1

                            cos_tensor = torch.tensor([cos])
                        else:

                            tmep = F.cosine_similarity(temp_seq[j], temp_seq[k], dim=0)

                            tmep_1 = torch.tensor([tmep])
                            cos_tensor = torch.cat([cos_tensor, tmep_1], 0)
                # 开始组合 一个对话的内容
                if j == 0:
                    temp_cos_tensor = torch.softmax(cos_tensor, 0)
                    tem_x = torch.cat([torch.tensor([0.0]), temp_cos_tensor], 0)
                    all_current_tensor.append(tem_x.unsqueeze(0))


                else:
                    temp_cos_tensor = torch.softmax(cos_tensor, 0)
                    begin = temp_cos_tensor[:j]
                    tem_A = torch.cat([begin, torch.tensor([0.0])], 0)
                    tem_A = torch.cat([tem_A, temp_cos_tensor[j:]], 0)
                    x = tem_A.unsqueeze(0)

                    all_current_tensor.append(x)

            bank_s_list.append(all_current_tensor)


        return bank_s_list
            # if i ==0:
            #     all_tensor = all_current_tensor.unsqueeze(0)
            # else:
            #     cos_tensor = cos_tensor.unsqueeze(0)
            #     all_tensor = torch.cat([all_tensor,cos_tensor],0)


    def forward(self, U_s, U_p, seq_lengths,speaker_cos_index,speaker_local_index):
        # (b) <== (l,b,h)
        batch_size = U_s.size(1)
        batch_index, context_s_, context_p_ = [], [], []
        for j in range(batch_size):

            batch_index.extend([j] * seq_lengths[j])
            context_s_.append(U_s[:seq_lengths[j], j, :])
            context_p_.append(U_p[:seq_lengths[j], j, :])

        # 计算context相似度


        batch_index = torch.tensor(batch_index)
        # bank_s_：[1653， 200]
        bank_s_ = torch.cat(context_s_, dim=0)
        bank_p_ = torch.cat(context_p_, dim=0)
        if self.cuda_flag:
            batch_index = batch_index.cuda()
            bank_s_ = bank_s_.cuda()
            bank_p_ = bank_p_.cuda()

        # (l,b,h) << (l*b,h)
        bank_s, bank_p = feature_transfer(bank_s_, bank_p_, seq_lengths, self.cuda_flag)
        bank_s_c=bank_s
        bank_p_c=bank_p
        sum = 0
        bank_s_c =  bank_s_c.transpose(1,0)
        #count = 0
        all_tensor = torch.empty(len(seq_lengths))

        bank_s_list = self.make_cos(bank_s,seq_lengths)


        bank_s_ = self.m_fc1(bank_s_)
        bank_p_ = self.m_fc2(bank_p_)
        #激活是只做一次 所以
        #循环体


        # start: 记录bank_s_中，每个conversation的开始位置
        input_conversation_length = torch.tensor(seq_lengths)
        # 开始的0，第一个conversation的开始位置
        start_zero = input_conversation_length.data.new(1).zero_()
        start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)  # cumsum表示前面的元素的累和

        # 处理situation-level
        feature_ = []
        # bank_s：[110，32，200]
        # speaker_out = self.SpeakerReasonModel(speaker_cos_index,speaker_local_index,q_star)
        #开始将其进行补齐操作补充到长度的位置
        convers_length = bank_s.size(0)
        flag_conver = True
        tensor_bank_s_list_1 = []
        tensor_bank_s_list = bank_s_list[0]
        tensor_bank_s_list = torch.cat(tensor_bank_s_list,0)
        tensor_bank_s_list = pad(tensor_bank_s_list, convers_length, cuda_flag=self.cuda_flag).unsqueeze(0)
        for i in bank_s_list:

                i= torch.cat(i,0)
                i= pad(i, convers_length, cuda_flag=self.cuda_flag)
                tensor_bank_s_list_1.append(i)


        for t in range(bank_s.size(0)):  # max_len，每个conversation的第t个utterance
            # (2*h) <== (h)
            # bank_s[t]：所有conversation的第t句u话
            q_star = self.q_fc(bank_s[t])  # fc扩充维度

            q_situ = self.reason_modules[0](bank_s_, batch_index, q_star,tensor_bank_s_list_1,bank_s,t,self.cuda_flag)
            feature_.append(q_situ.unsqueeze(0))
            # 推理结果编码到长时记忆
            indexs = []
            values = []
            for i in range(bank_s[t].size(0)):  # 第i个conversation
                if t < seq_lengths[i]:
                    index = start[i] + t
                    q_situ_1= self.w1_1(q_situ[i])
                    # q_situ_1 = self.dropout(q_situ_1)
                    q_situ_2= self.w1_2(bank_s_[index])
                    # q_situ_2 = self.dropout(q_situ_2)
                    z_1 = torch.tanh(q_situ_1+q_situ_2)
                    q_situ_update = z_1*bank_s_[index]

                    indexs.append(index)
                    values.append(q_situ_update)
            indices = [torch.LongTensor(indexs)]
            values = torch.stack(values, 0)
            bank_s_ = bank_s_.index_put(indices=indices, values=values)

        feature_s = torch.cat(feature_, dim=0)

        # 已经获取到了对应位置下标的内容 开始对其进行填充工作

        bank_p_list = []
        # 处理speaker-level
        # speaker_cos_index
        #
        # speaker_local_index
        #当前已经保存好了speaker的信息 每次只需要对当前进行处理
        feature_ = []
        length = feature_s.size(0)
        speaker_output = self.SpeakerReasonModel(speaker_cos_index,speaker_local_index,bank_s,length,self.cuda_flag)
        trans_speaker_output = speaker_output.transpose(0,1)

        # (l,b,2*2*h)
        hidden = torch.cat([trans_speaker_output, feature_s], dim=-1)
        hidden_result = self.dropout(F.relu(hidden))
        log_prob = F.log_softmax(self.smax_fc(hidden_result), 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob


class DialogueCRN(nn.Module):
    def __init__(self, base_model='LSTM', base_layer=2, input_size=None, hidden_size=None, n_speakers=2,
                 n_classes=7, dropout=0.2, cuda_flag=False, reason_steps=None):
        """
        Contextual Reasoning Network
        """
        # print(reason_steps)
        super(DialogueCRN, self).__init__()
        self.base_model = base_model
        self.n_speakers = n_speakers

        if self.base_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer, bidirectional=True, dropout=dropout)
            self.rnn_parties = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer, bidirectional=True, dropout=dropout)
        elif self.base_model == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer, bidirectional=True, dropout=dropout)
            self.rnn_parties = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=base_layer, bidirectional=True, dropout=dropout)
        elif self.base_model == 'Linear':
            self.base_linear = nn.Linear(input_size, 2 * hidden_size)
        else:
            print('Base model must be one of LSTM/GRU/Linear')
            raise NotImplementedError

        self.cognition_net = CognitionNetwork(n_features=2 * hidden_size, n_classes=n_classes, dropout=dropout, cuda_flag=cuda_flag, reason_steps=reason_steps)
        print(self)

    def make_cos_person_X(self, bank_s_c):
        # bank_s_c = bank_s_c.transpose(0,1)
        temp_seq = bank_s_c
        all_current_tensor = []
        for j in range(len(bank_s_c)):
            count = 0
            for k in range(len(bank_s_c)):
                if j == k:
                    pass
                else:
                    if count == 0:
                        cos = F.cosine_similarity(temp_seq[j], temp_seq[k], dim=0)

                        count =count+ 1

                        cos_tensor = torch.tensor([cos])#方括号是变成（1,100）的形式
                    else:
                        tmep = F.cosine_similarity(temp_seq[j], temp_seq[k], dim=0)

                        tmep_1 = torch.tensor([tmep])
                        cos_tensor = torch.cat([cos_tensor, tmep_1], 0)
            # 开始组合 一个对话的内容
            if j == 0:
                temp_cos_tensor = torch.softmax(cos_tensor, 0)
                tem_x = torch.cat([torch.tensor([0.0]), temp_cos_tensor], 0)
                all_current_tensor.append(tem_x.unsqueeze(0))


            else:
                temp_cos_tensor = torch.softmax(cos_tensor, 0)
                begin = temp_cos_tensor[:j]
                tem_A = torch.cat([begin, torch.tensor([0.0])], 0)
                tem_A = torch.cat([tem_A, temp_cos_tensor[j:]], 0)
                x = tem_A.unsqueeze(0)

                all_current_tensor.append(x)


        return all_current_tensor

    def forward(self, U, qmask, seq_lengths):
        U_s, U_p = None, None
        if self.base_model == 'LSTM':

            U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]

            for b in range(U_.size(0)):  # b表示一个conversation
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:

                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]

            E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]

            speaker_local_index = []
            speaker_cos_index = []
            flage = False

            for b in range(U_p_.size(0)):
                index_array = torch.empty(2)
                utterence_dir = {}
                utterence_cos_dir = {}
                for p in range(len(U_parties_)):

                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                     #首先获取到对应下标的句子表示该speaker的global memory 代表的内容
                    #先保存好该speaker 的内容 然后根据 相似度计算出归一划的结果
                    #计算出结果后保存，并赋予一个对应的global 下标位置信息 以供后面查询
                    #如下 存储的应该是[batch utterence speaker_index]
                    #cos 的相似度结果应该需要保存的结果为[batch utterence cos_sim]
                    #使用字典进行存储对应位置下的内容 这样可以避免存在长度不一致的问题
                    count = 0
                    for i in index_i:
                        if count == 0:
                            speaker_temp = E_parties_[p][b][i].unsqueeze(0)
                            count = count+1
                        else:
                            temp = E_parties_[p][b][i].unsqueeze(0)
                            speaker_temp = torch.cat([speaker_temp,temp],0)
                    #求出每个人的全部rnn输出
                    temp_utterence_cos = self.make_cos_person_X(speaker_temp)
                    count_cos = 0

                    for i in index_i:
                        t = int(i)
                        utterence_cos_dir[t] = temp_utterence_cos[count_cos]
                        count_cos=count_cos+1
                        #首先得到一个global speaker 内容
                        utterence_dir[t] =speaker_temp
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
                speaker_cos_index.append(utterence_cos_dir)
                speaker_local_index.append(utterence_dir)


            U_p = U_p_.transpose(0, 1)

            # (l,b,2*h) [(2*bi,b,h) * 2]
            U_s, hidden = self.rnn(U)

        elif self.base_model == 'GRU':
            U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], 200).type(U.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]

            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)
            U_s, hidden = self.rnn(U)
        elif self.base_model == 'None':
            # TODO
            U_s = self.base_linear(U)
        logits = self.cognition_net(U_s, U_p, seq_lengths,speaker_cos_index,speaker_local_index)
        return logits

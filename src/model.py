import torch
import torch.nn as nn

class RefiedKBQA(nn.Module):
    """
    Define the neural model to calculate relation
    """
    def __init__(self, N_W2V, N_R, kb_info):
        """Initialize the model
        Args:
            N_W2V(int): number of dim in word2vec  
            N_R(int): number of relations
            kb_info: M_subj, M_rel, M_obj matrixes where
                M_subj(matrix): dim (N_T, N_E) where N_T is the number of
                                 triples in the KB
                M_rel(matrix): dim (N_T, N_R)
                M_obj(matrix): dim (N_T, N_E)
        """
        super().__init__()
        self.dense1 = nn.Linear(N_W2V, N_R) # for the 1 hop
        self.dense2 = nn.Linear(N_W2V, N_R) # for the 2 hop
        self.dense3 = nn.Linear(N_W2V, N_R) # for the 3 hop
        self.M_subj, self.M_rel, self.M_obj = kb_info # store the kb info

    def _one_step(self, x, r):
        """One step follow
        Args:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
            r(matrix): batched relation embeddings corresponding to
                        the x with dim (batch_size, N_R)

        Return:
            x_new(matrix): batched k-hot vector with dim (batch_size, N_E)
                            where N_E is the number of relations
        """
        # vector x * M_subj^T
        print(self.M_subj.size(), x.size())
        x_t = torch.sparse.mm(self.M_subj, x.T) # (N_T, batch_size) dense
        #x_t = torch.mm(x, self.M_subj.T)
        # vector r * M_subj^T
        r_t = torch.sparse.mm(self.M_rel.T, r.T) # (N_T, batch_size) dense
        #r_t = torch.mm(r, self.M_rel.T)
        # (x_t * r_t) * M_obj
        x_new = torch.sparse.mm(self.M_obj.T, x_t * r_t).T # (batch_size, N_E)
        #x_new = torch.mm(x_t * r_t, self.M_obj)
        return x_new

    def forward(self, x, q, n_hop):
        """Forward process
        Args:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
            q(matrix): batched question embeddings corresponding to
                        the x with dim (batch_size, N_W2V)

        Return:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
        """
        # calculate r for each hop
        # r = q^T w
        if 1 <= n_hop:
            r = self.dense1(q)
            x = self._one_step(x, r)
        if 2 <= n_hop:
            r = self.dense2(q)
            x = self._one_step(x, r)
        if 3 <= n_hop:
            r = self.dense3(q)
            x = self._one_step(x, r)

        return x

class RefiedKBQA_KBCompletion(nn.Module):
    """
    Define the neural model to calculate relation
    """
    def __init__(self, N_W2V, N_R, kb_info):
        """Initialize the model
        Args:
            N_W2V(int): number of dim in word2vec  
            N_R(int): number of relations
            kb_info: M_subj, M_rel, M_obj matrixes where
                M_subj(matrix): dim (N_T, N_E) where N_T is the number of
                                 triples in the KB
                M_rel(matrix): dim (N_T, N_R)
                M_obj(matrix): dim (N_T, N_E)
        """
        super().__init__()
        self.dense1 = nn.Linear(N_W2V, N_R) # for the 1 hop
        self.dense2 = nn.Linear(N_W2V, N_R) # for the 2 hop
        self.dense3 = nn.Linear(N_W2V, N_R) # for the 3 hop
        self.M_subj, self.M_rel, self.M_obj = kb_info # store the kb info

    def _one_step(self, x, r):
        """One step follow
        Args:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
            r(matrix): batched relation embeddings corresponding to
                        the x with dim (batch_size, N_R)

        Return:
            x_new(matrix): batched k-hot vector with dim (batch_size, N_E)
                            where N_E is the number of relations
        """
        # vector x * M_subj^T
        x_t = torch.dot(x, self.M_subj.T)
        # vector r * M_subj^T
        r_t = torch.dot(r, self.M_rel.T)
        # (x_t * r_t) * M_obj
        x_new = torch.dot(x_t * r_t, self.M_obj) + x
        return x_new

    def forward(self, x, q, n_hop):
        """Forward process
        Args:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
            q(matrix): batched question embeddings corresponding to
                        the x with dim (batch_size, N_W2V)

        Return:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
        """
        # calculate r for each hop
        # r = q^T w
        if 1 <= n_hop:
            r = self.dense1(q)
            x = self._one_step(x, r)
        if 2 <= n_hop:
            r = self.dense2(q)
            x = self._one_step(x, r)
        if 3 <= n_hop:
            r = self.dense3(q)
            x = self._one_step(x, r)

        return x

class LSTM_Encoder(nn.Module):
    """
    Define the encoder for the encoder-decoder model
    """
    def __init__(self, input_size, hidden_size):
        """Initialize the model
        Args:
            input_size(int): the dimension of the input vector for each token
                                which is N_W2V
            hidden_size(int): the dimension of the model hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        
    def forward(self, x):
        """Forward process
        Args:
            x(matrix): batched embeded question matrix with size
                        (sequence length, batch size, N_W2V)

        Return:
            hidden(tuple): (hn, cn) where both have size (1, batch size, hidden_size)
        """
        lstm_out, self.hidden = self.lstm(x)
        return self.hidden

class LSTM_Decoder(nn.Module):
    """
    Define the encoder for the encoder-decoder model
    """
    def __init__(self, input_size, hidden_size, N_W2V):
        """Initialize the model
        Args:
            input_size(int): the dimension of the input vector for each realtion
                                which is N_R
            hidden_size(int): the dimension of the model hidden state
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.dense_p = nn.Linear(hidden_size, 1) # for the probability of stopping
        self.sigmoid = nn.Sigmoid() # for p
        self.dense_r = nn.Linear(hidden_size, input_size) # for the relation vector
        self.softmax = nn.Softmax() # for r
        
    def forward(self, x, hidden):
        """Forward process
        Args:
            x(matrix): batched embeded relation hop matrix with size
                        (hop length, batch size, N_R). Suppose the embedding
                        for each relation is a one-hot vector
            hidden(tuple): (hn, cn) where both have size (1, batch size, hidden_size)

        Return:
            ps(matrix): output of p with size (hop length, batch size, 1)
            rs(matrix): output of relation with size (hop length, batch size, N_R)
        """
        # lstm_out(matrix): batched output of all states with size
        #                     (hop length, batch size, hidden_size)
        lstm_out, self.hidden = self.lstm(x, hidden)
        ps = self.sigmoid(self.dense_p(lstm_out)) # stoppings
        rs = self.softmax(self.dense_r(lstm_out)) # relations
        return ps, rs

class RefiedKBQA_LSTM(nn.Module):
    """
    Define the encoder-decoder neural model to calculate relation
    """
    def init(self, N_W2V, N_R, hidden_size, kb_info):
        """Initialize the model
        Args:
            N_W2V(int): number of dim in word2vec  
            N_R(int): number of relations
            kb_info: M_subj, M_rel, M_obj matrixes where
                M_subj(matrix): dim (N_T, N_E) where N_T is the number of
                                 triples in the KB
                M_rel(matrix): dim (N_T, N_R)
                M_obj(matrix): dim (N_T, N_E)
        """
        super().__init__()
        self.lstm_encoder = nn.LSTM_Encoder(input_size=N_W2V, hidden_size=hidden_size)
        self.lstm_decoder = nn.LSTM_Decoder(intput_size=N_R, hidden_size=hidden_size)
        self.M_subj, self.M_rel, self.M_obj = kb_info # store the kb info
        self.softmax = nn.Softmax() # for the weighted sum

    def _one_step(self, x, r):
        """One step follow
        Args:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
            r(matrix): batched relation embeddings corresponding to
                        the x with dim (batch_size, N_R)

        Return:
            x_new(matrix): batched k-hot vector with dim (batch_size, N_E)
                            where N_E is the number of relations
        """
        # vector x * M_subj^T
        x_t = torch.dot(x, self.M_subj.T)
        # vector r * M_subj^T
        r_t = torch.dot(r, self.M_rel.T)
        # (x_t * r_t) * M_obj
        x_new = torch.dot(x_t * r_t, self.M_obj) + x
        return x_new

    def forward(self, x, tokens, relations, n_hop):
        """Forward process
        Args:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
            tokens(matrix): batched question token embeddings corresponding to
                            the x with dim (sequence length, batch_size, N_W2V)
            relations(matrix): batched relation one-hot vector with size
                                (hop length, batch size, N_R)
                                ------ I assume the one-hot vector for the input

        Return:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
            stoppings(matrix): output of p with size (hop length, batch size, 1)
            relations(matrix): output of relation with size (hop length, batch size, N_R)
        """
        # calculate r for each hop
        hidden = self.lstm_encoder(tokens) # embedding for the q
        stoppings, relations = self.lstm_decoder(relations, hidden)
        non_stoppings = 1 - stoppings # to calculate 1-p
        non_stoppings = torch.cumprod(non_stoppings, dim=0) # cumulative product for t'<t
        non_stoppings = torch.roll(non_stoppings, 1, dim=0) # shift right for corresponding t
        non_stoppings[0] = torch.ones_like(non_stoppings[0].size()) # set the first column as 1

        # calculate x
        xs = []
        for i in range(relations.size(0)):
            r = relations[i]
            x = self._one_step(x, r)
            xs.append(x.unsqueeze(0))
        x = torch.cat(xs, dim=0) # concat all x from all hops, size (hop length, batch_size, N_E)
        x = x * stoppings * non_stoppings # x times weight that is made by stoppings
        x = self.softmax(torch.sum(x, dim=0)) # weighted sum of all x in all time
        return x, stoppings, relations

if __name__ == "__main__":
    RefiedDBQA(128, 100, None)
    RefiedDBQA_KBCompletion(128, 100, None)
    RefiedDBQA_LSTM(128, 100, None)

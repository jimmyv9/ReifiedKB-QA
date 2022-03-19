import torch
import torch.nn as nn

class RefiedKBQA(nn.Module):
    """
    Define the neural model to calculate relation
    """
    def init(self, N_W2V, N_R):
        """Initialize the model
        Args:
            N_W2V(int): number of dim in word2vec  
            N_R(int): number of relations
        """
        self.dense1 = nn.Linear(n_w2v, n_r) # for the 1 hop
        self.dense2 = nn.Linear(n_w2v, n_r) # for the 2 hop
        self.dense3 = nn.Linear(n_w2v, n_r) # for the 3 hop

    def forward(self, x, q, n_hop, M_subj, M_rel, M_obj):
        """Forward process
        Args:
            x(matrix): batched k-hot vector with dim (batch_size, N_E)
                        where N_E is the number of relations
            q(matrix): batched question embeddings corresponding to
                        the x with dim (batch_size, N_W2V)
            M_subj(matrix): dim (N_T, N_E) where N_T is the number of
                             triples in the KB
            M_rel(matrix): dim (N_T, N_R)
            M_obj(matrix): dim (N_T, N_E)

        Return:
            x_new(matrix): batched k-hot vector with dim (batch_size, N_E)
                            where N_E is the number of relations
        """
        # calculate r for each hop
        # r = q^T w
        if 1 == n_hop:
            r = self.dense1(q)
        if 2 == n_hop:
            r = self.dense2(q)
        if 3 == n_hop:
            r = self.dense3(q)

        # vector x * M_subj^T
        x_t = torch.dot(x, M_subj.T)
        # vector r * M_subj^T
        r_t = torch.dot(r, M_rel.T)
        # (x_t * r_t) * M_obj
        x_new = torch.dot(x_t * r_t, self.M_obj)

        return x_new

if __name__ == "__main__":
    RefiedDBQA(128, 100, None, None, None)

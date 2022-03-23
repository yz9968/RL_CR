import torch
import torch.nn as nn
import torch.nn.functional as F
		
class Encoder(nn.Module):
	def __init__(self, din=32, hidden_dim=128):
		super(Encoder, self).__init__()
		self.fc = nn.Linear(din, hidden_dim)

	def forward(self, x):
		embedding = F.relu(self.fc(x))
		return embedding

class AttModel(nn.Module):
	def __init__(self, n_node, din, hidden_dim, dout):
		super(AttModel, self).__init__()
		self.fcv = nn.Linear(din, hidden_dim)
		self.fck = nn.Linear(din, hidden_dim)
		self.fcq = nn.Linear(din, hidden_dim)
		# self.fcout = nn.Linear(hidden_dim, dout)

	def forward(self, x, mask):
		v = F.relu(self.fcv(x))
		q = F.relu(self.fcq(x))
		k = F.relu(self.fck(x)).permute(0,2,1)
		# h = torch.clamp(torch.mul(torch.bmm(q,k), mask), 0 , 9e13) - 9e15*(1 - mask)
		# h = torch.mul(torch.bmm(q,k), mask)
		h = mask.view_as(torch.mul(torch.bmm(q,k), mask))
		h = h.where(h!=0,torch.full(h.shape,-9e5).cuda())
		att = F.softmax(h, dim=2)

		out = torch.bmm(att,v)
		#out = torch.add(out,v)
		#out = F.relu(self.fcout(out))
		return out, h

class Q_Net(nn.Module):
	def __init__(self, hidden_dim, dout):
		super(Q_Net, self).__init__()
		self.fc = nn.Linear(hidden_dim, dout)

	def forward(self, x):
		q = self.fc(x)
		return q

class DGN(nn.Module):
	def __init__(self,n_agent,num_inputs,hidden_dim,num_actions):
		super(DGN, self).__init__()
		
		self.encoder = Encoder(num_inputs,hidden_dim)
		self.att = AttModel(n_agent,hidden_dim,hidden_dim,hidden_dim)
		self.q_net = Q_Net(hidden_dim,num_actions)
		
	def forward(self, x, mask):
		h1 = self.encoder(x)
		h2, a_w = self.att(h1, mask)
		q = self.q_net(h2)
		return q, a_w

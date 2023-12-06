###########################################################
#
# ShakeGPT
#
# Learning to emit endless Shakespeare to demonstrate how
# decoder-only next-token prediction transformer models 
# work
#
#
# Luke Sheneman
# sheneman@uidaho.edu
# December 2023
#
# Research Computing and Data Services (RCDS)
# Insitite for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
#
##########################################################

import torch
from torch.nn import functional as F
import torch.nn as nn
import time

INPUT_CORPUS     = "complete_shakespeare.txt"
OUTPUT_FILE      = "AI_shakespeare_out.txt"

SEED             = int(time.time())
TRAIN_SIZE       = 0.85
VAL_SIZE         = 1.0 - TRAIN_SIZE
BATCH_SIZE       = 200
NUM_BATCHES      = 12000
LEARNING_RATE    = 3e-4
CONTEXT_WINDOW   = 32 
EMBEDDING_SIZE   = 64
NUM_LAYERS       = 6
NUM_HEADS        = 8
FFN_HIDDEN_SIZE  = 8
LOSS_SAMPLE_SIZE = 50





shakespeare = """
   ⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣠⣄⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀
   ⠀⠀⠀⠀⠀⢀⣴⠟⠛⠉⠉⠉⠉⠛⠻⣦⡀⠀⠀⠀⠀⠀⠀
   ⠀⠀⠀⠀⢰⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⡆⠀⠀⠀⠀⠀
   ⠀⠀⠀⠀⣼⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣷⡀⠀⠀⠀⠀
   ⠀⠀⠀⣰⣿⣿⣿⣤⣤⣄⠀⠀⣠⣤⣤⣿⣿⣿⣷⡀⠀⠀⠀
   ⠀⢀⣼⣿⣿⣿⠋⢠⣤⠙⠁⠈⠋⣤⡄⠙⣿⣿⣿⣿⣄⠀⠀
   ⢠⣿⣿⣿⣿⡿⠀⠈⠉⠀⠀⠀⠀⠉⠁⠀⢿⣿⣿⣿⣿⣷⠀
   ⣿⣿⣿⣿⣿⣇⠀⠀⠀⠀⡀⢀⠀⠀⠀⠀⣸⣿⣿⣿⣿⣿⡆
   ⠹⣿⣿⣿⣿⣿⠀⠀⠴⠞⠁⠈⠳⠦⠀⠀⣿⣿⣿⣿⣿⡿⠁
   ⠀⠉⢻⡿⢿⣿⣧⠀⠀⠀⢶⡶⠀⠀⠀⣼⣿⣿⣿⡟⠋⠁⠀
   ⠀⠀⣼⡇⠀⠀⠙⣷⣄⠀⠈⠁⠀⣠⣾⠋⠀⠀⢸⣧⠀⠀⠀
   ⠀⠀⣿⡇⠀⠀⠀⠈⠛⠷⣶⣶⠾⠛⠁⠀⠀⠀⢸⣿⠀⠀⠀
   ⠀⠀⢻⡇⠀⠀⠀⣀⣀⣤⣤⣤⣤⣀⣀⠀⠀⠀⢸⡟⠀⠀⠀
   ⠀⠀⠘⣿⣴⠾⠛⠋⠉⠉⠉⠉⠉⠉⠛⠛⠷⣦⣿⠃⠀⠀⠀
   ⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀
All your word are belong to us!

     Shakespeare AI @ UI

"""

print(shakespeare)





#
# Set seed and device (GPU or CPU)
#
torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("DEVICE=", device)
print("\n")


#
# Read the Complete Works of Shakespeare into a single string
#
# From:  Project Gutenburg
# (wget https://www.gutenberg.org/ebooks/100.txt.utf-8)
#
print("Reading input...", flush=True)
with open(INPUT_CORPUS, 'r', encoding='utf-8') as f:
    text = f.read()

vocab      = list(set(text))
vocab_size = len(vocab)
print("Detected vocabulary of size %d in corpus." %vocab_size, flush=True)


#
# Tokenizer
#
char_to_idx  = { ch:i for i,ch in enumerate(vocab) }
idx_to_char  = { i:ch for i,ch in enumerate(vocab) }
string2tokens = lambda string: [char_to_idx[character] for character in string] 
tokens2string = lambda tokens: ''.join([idx_to_char[index] for index in tokens])

tokenized_corpus = torch.tensor(string2tokens(text), dtype=torch.long)


#
# Derive our TEST and VALIDATION sets from the tokenized corpus
#
train_portion = int(TRAIN_SIZE*len(tokenized_corpus)) 
training_data   = tokenized_corpus[:train_portion]
validation_data = tokenized_corpus[train_portion:]


#
# Extract a batch of size BATCH_SIZE from either training or validation sets
#
def batch(data):
    selected = torch.randint(len(data) - CONTEXT_WINDOW, (BATCH_SIZE,))
    input    = torch.stack([data[i:i+CONTEXT_WINDOW] for i in selected])
    target   = torch.stack([data[i+1:i+CONTEXT_WINDOW+1] for i in selected])

    return input.to(device), target.to(device)



#
# Evaluate the model during training to estimate average loss for the 
# given dataset (training or validation)
#
def sample_loss(data):

    loss_tensor = torch.zeros(LOSS_SAMPLE_SIZE)
    for i in range(LOSS_SAMPLE_SIZE):
        input, target = batch(data)
        _,loss = model(input, target)
        loss_tensor[i] = loss.item()

    model.train()
    return(loss_tensor.mean())
 



#
# ATTENTION HEAD definition - Here is where the magic happens
#
class AttentionHead(nn.Module):

    def __init__(self, headspace):
        super().__init__()

        # The Query(Q), Key(K), and Value(V) vectors are just linear tensors
        self.query = nn.Linear(EMBEDDING_SIZE, headspace)
        self.key   = nn.Linear(EMBEDDING_SIZE, headspace)
        self.value = nn.Linear(EMBEDDING_SIZE, headspace)

        # Causal Self-Attention:  define our triangular mask so we can't look into the FUTURE...
        lower_diag = torch.tril(torch.ones(CONTEXT_WINDOW, CONTEXT_WINDOW))
        self.register_buffer('causal_mask', lower_diag)


    def forward(self, x):
        batchsize, seqlen, embedlen  = x.shape
        query_vector = self.query(x) 
        key_vector   = self.key(x)  

        attention_weights = query_vector @ key_vector.transpose(-2,-1) * torch.sqrt(torch.tensor(embedlen)) 
        attention_weights = attention_weights.masked_fill(self.causal_mask[:seqlen, :seqlen] == 0, float('-inf')) 
        attention_weights = F.softmax(attention_weights, dim=-1) 

        # weighted sum of value vector
        value_vector = self.value(x) 
        x = attention_weights @ value_vector

        return x



#
# MULTI-HEAD ATTENTION - Just a bunch of heads paying attention to different things.
#
class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        # define NUM_HEADS attention heads, each head able to attend to only an equalsubset of the embedding layer
        headspace = EMBEDDING_SIZE // NUM_HEADS
        self.heads  = nn.ModuleList([AttentionHead(headspace) for h in range(NUM_HEADS)])
        self.linear = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

    def forward(self, x):

        # concatenate the output from all heads and aggregate through a single linear layer
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.linear(x)

        return x




#
# Define FEED FORWARD NETWORK (FFN) portion of the TransformerBlock
#
class TransformerFeedFoward(nn.Module):

    def __init__(self):

        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE*FFN_HIDDEN_SIZE),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(EMBEDDING_SIZE*FFN_HIDDEN_SIZE, EMBEDDING_SIZE),
        )

    def forward(self, x):
        x = self.feed_forward(x)

        return(x)




#
# Define the TRANSFORMER BLOCK component, consisting of:
#
#    * Multiple Self-Attention Heads
#    * Feed-Forward neural networks
#    * Repeating Linear Layer Normalization layers
#
class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.self_attention = MultiHeadAttention()
        self.feed_forward   = TransformerFeedFoward()
        self.linear_norm    = nn.LayerNorm(EMBEDDING_SIZE)

    def forward(self, x):

        lnorm0 = self.linear_norm(x)
        x = x + self.self_attention(lnorm0)

        lnorm1 = self.linear_norm(x)
        x = x + self.feed_forward(lnorm1)

        return x



class ShakeGPT(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_embedding_table    = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        self.position_embedding_table = nn.Embedding(CONTEXT_WINDOW, EMBEDDING_SIZE)
        self.blocks                   = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        self.layernorm                = nn.LayerNorm(EMBEDDING_SIZE) # final layer norm
        self.lm_head                  = nn.Linear(EMBEDDING_SIZE, vocab_size)

    def forward(self, input, targets=None):
        batchsize, seqlen = input.shape

        tok_emb = self.token_embedding_table(input) 
        pos_emb = self.position_embedding_table(torch.arange(seqlen, device=device)) 
        x       = tok_emb + pos_emb
        x       = self.blocks(x) 
        x       = self.layernorm(x) 
        logits  = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            batchsize, seqlen, embedlen = logits.shape
            logits  = logits.view(batchsize*seqlen, embedlen)
            targets = targets.view(batchsize*seqlen)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, length):
        for i in range(length):
            idxc         = idx[:, -CONTEXT_WINDOW:]
            logits,_     = self(idxc)
            logits       = logits[:, -1, :]
            probs        = F.softmax(logits, dim=-1) 
            idx_next     = torch.multinomial(probs, num_samples=1)
            idx          = torch.cat((idx, idx_next), dim=1) 

        return idx



#
# Instantiate our PyTorch transformer model and send it to the GPU
#
model     = ShakeGPT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)




#
# *** TRAINING ***
#
# Train in batches of size BATCH_SIZE for NUM_BATCHES
# Occasionally evaluate the current model to determine training and validation loss
#
for b in range(NUM_BATCHES):

    # Pull a random training batch
    input, target = batch(training_data)

    # Forward pass of data through the model
    _,loss = model(input, target)

    optimizer.zero_grad() # Reset our gradients for this batch
    loss.backward()       # Backpropagation
    optimizer.step()      # Neural Network Optimization

    # Every 100 batches, let's sample and report our loss
    if(not b%100):
        torch.no_grad()
        model.eval()
        train_loss = sample_loss(training_data)
        val_loss   = sample_loss(validation_data)
        print(f"BATCH %s/%d: training loss=%.05f | validation loss=%.05f" %(str(b).zfill(4),NUM_BATCHES,train_loss,val_loss))
        model.train()





#
# *** INFERENCE ***
#
# Training done.  Let's spew a chunk of Shakespearish
#
print("INFERENCE TIME.  SPEWING...(be patient)", flush=True)

f = open(OUTPUT_FILE, "w", encoding="utf-8")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
x = model.generate(context, length=5000)
generated_text = tokens2string(x[0].tolist())
print(generated_text)
f.write(generated_text)
f.close()



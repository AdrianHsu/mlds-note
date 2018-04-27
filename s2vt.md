# Step-by-step S2VT (seq2seq) Implementation



## Input

`(80, 4096)` video features, extract from VGG19, of which `80` is the number of frames selected from videos.



## My use words

* text: "I love you."
* caption: `["I", "love", "you"]`
* sequence: `[1, 3, 2]`, that is, transform those 2890 different vocab into one index. from 4 to 2890. ( 0~3 is for special tokens)



## Global variables

```Python
filters='`","?!/.()'
n_inputs = 4096 # dimension of the input video features
n_hidden_red = 128 # first layer lstm (lstm_red), num of units
n_hidden_gre = 256 # second layer lstm (lstm_gre), num of units
embedding_size = 128 # embed 2890 different vocab (e.g. and, girl) into 500 dim vector 
max_time = 80 # enc: 80 frames
max_caption_len = 70 # dec: to pad the len into same size for all captions
sequence_length = max_time + max_caption_len # this is the max time steps of lstm-red 
word_min_counts_threshold = 3
forget_bias_red = 1.0
forget_bias_gre = 1.0
dropout_prob = 0.5

special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
special_tokens_to_word = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
```



## Class DataObject

```python
class DataObject:
    def __init__(self, path, myid, dat, caption_list, cap_len_list):
        self.path = path # feat_dir + myid + '.npy'
        self.myid = myid # 'xBePrplM4OA_6_18.avi'
        self.dat = dat # npy load-in martrix
        self.caption_list = caption_list # no EOS!!, e.g. ['I', 'love', 'you']
        self.cap_len_list = cap_len_list # EOS added, e.g. 4
```



# Class Dataset

```python
class Dataset:
    def __init__(self, feat_dir, corpus_dir, json_filename, max_size):
        self.feat_dir = feat_dir # /home/data/MLDS_hw2_1_data/training_data/feat/
        self.json_filename = json_filename # training_label.json OR testing_labl.json
        self.corpus_dir = corpus_dir # /home/data/MLDS_hw2_1_data/, where .json puts here
        self.json_train_filename = 'training_label.json'
        self.data_obj_list = [] # all DataObjects
        self.word_min_counts_threshold = word_min_counts_threshold
        self.vocab_num = 0
        self.word_counts = {} # every words appear time
        self.word_index = {} # word_to_id
        self.idx_to_word = {}

        self.batch_max_size = max_size # 1450: train, 100: test
        self.batch_index = 0 # we need cuurent index if we sequentially read next batch 

       
    def sample_one_caption(self, captions, cap_len, is_rand=True):

        assert len(captions) == len(cap_len)
        if is_rand:
            r = np.random.randint(0, len(captions))
        else:
            r = 0
        return captions[r], cap_len[r]

    def captions_to_padded_sequences(self, captions, maxlen=max_caption_len):

        res = []
        for cap in captions:
            l = []
            for word in cap:
                if word in self.word_counts:
                    l.append(self.word_index[word])
                else:
                    l.append(special_tokens['<UNK>'])
            l.append(special_tokens['<EOS>']) # add EOS here!

            pad = special_tokens['<PAD>']
            l += [ pad ] * (maxlen - len(l))

            res.append(l)
        return res
    
    def schedule_sampling(self, sampling_prob):
        sampling = np.zeros((FLAGS.batch_size, max_caption_len), dtype = bool)
        for b in range(FLAGS.batch_size):
            if np.random.uniform(0,1,1) < sampling_prob:
                for l in range(max_caption_len):
                    sampling[b,l] = True
            else:
                for l in range(max_caption_len):
                    sampling[b,l] = False
        sampling[0,:] = True # To ensure <BOS> would 100% be selected... 

        return sampling

    def next_batch(self, is_rand=True): 
        
        if not is_rand or FLAGS.test_mode:
            # 1. sequential chosen
            current_index = self.batch_index
            max_size = self.batch_max_size
            if current_index + FLAGS.batch_size <= max_size:
                dat_list = self.data_obj_list[current_index:(current_index + FLAGS.batch_size)]
                self.batch_index += FLAGS.batch_size
            else:
                right = FLAGS.batch_size - (max_size - current_index)
                dat_list = self.data_obj_list[current_index:max_size] + self.data_obj_list[0: right]
                self.batch_index = right
        else:
            # 2. random chosen
            indices = random.sample(range(self.batch_max_size), FLAGS.batch_size)
            dat_list = [self.data_obj_list[i] for i in sorted(indices)]
        
        img_batch = []
        cap_batch = []
        id_batch = []
        cap_len = []
        for d in dat_list:
            img_batch.append(d.dat)
            id_batch.append(d.myid)
            cap, l = self.sample_one_caption(d.caption_list, d.cap_len_list)
            cap = np.array(cap)
            cap_batch.append(cap)
            cap_len.append(l)
        cap_batch = self.captions_to_padded_sequences(cap_batch)
        img_batch = np.array(img_batch)
        id_batch = np.array(id_batch)
        cap_batch = np.array(cap_batch)
        cap_len = np.array(cap_len)
        return img_batch, cap_batch, cap_len, id_batch

    def prep_token_list(self):
        corpus_path = self.corpus_dir + self.json_train_filename
        train_file = pd.read_json(corpus_path)
        total_list = []
        for i in range(0, len(train_file['caption'])):
            str_list = train_file['caption'][i]
            for j in range(0, len(str_list)):
                total_list.append(str_list[j])
        return total_list

    def prep_tokenizer(self):
        total_list = self.prep_token_list()

        tokenizer = Tokenizer(filters=filters, lower=True, split=" ")
        tokenizer.fit_on_texts(total_list)

        for tok in tokenizer.word_counts.items():
            if tok[1] >= self.word_min_counts_threshold:
                self.word_counts[tok[0]] = tok[1]

        self.vocab_num = len(self.word_counts) + 4 # init vocab_num, must add 4 special tokens!!

        for i in range(0, 4):
            tok = special_tokens_to_word[i]
            self.word_index[tok] = i
            self.idx_to_word[i] = tok

        cnt = 0
        for tok in tokenizer.word_index.items():
            if tok[0] in self.word_counts:
                self.word_index[tok[0]] = cnt + 4
                self.idx_to_word[cnt + 4] = tok[0]
                cnt += 1
        
        
        #assert len(self.word_counts) == self.vocab_num # no!! they are not equal
        assert len(self.word_index) == self.vocab_num # yes! they are equal
        
        return self.vocab_num # for embedding 

    def build_data_obj_list(self):
        corpus_path = self.corpus_dir + self.json_filename

        data_file = pd.read_json(corpus_path)
        for i in range(0, len(data_file['caption'])):

            myid = data_file['id'][i]
            path = self.feat_dir + myid + '.npy'
            mydat = np.load(path)
            str_list = data_file['caption'][i]
            tmp_list = []
            cap_len_list = []
            for j in range(0, len(str_list)):
                seq = text_to_word_sequence(str_list[j], filters=filters, 
                    lower=True, split=" ")
                tmp_list.append(seq)
                #cap_len_list.append(len(seq) + 1) # added <EOS>
                if FLAGS.test_mode:
                    cap_len_list.append(len(seq) + 1) # added <EOS>
                else:
                    cap_len_list.append(max_caption_len) # added <EOS>

            obj = DataObject(path, myid, mydat, tmp_list, cap_len_list)
            self.data_obj_list.append(obj)
```










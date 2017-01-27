import dynet
import argparse
import util
import time
import random
import sys
import seq2seq

# input args

parser = argparse.ArgumentParser()

## need to have this dummy argument for dynet
parser.add_argument("--dynet-mem")
parser.add_argument("--dynet-gpu")

## locations of data
parser.add_argument("--train")
parser.add_argument("--valid")
parser.add_argument("--test")

## alternatively, load one dataset and split it
parser.add_argument("--percent_valid", default=1000, type=float)

## vocab parameters
parser.add_argument('--rebuild_vocab', action='store_true')
parser.add_argument('--unk_thresh', default=1, type=int)

## rnn parameters
parser.add_argument("--layers", default=1, type=int)
parser.add_argument("--input_dim", default=256, type=int)
parser.add_argument("--hidden_dim", default=256, type=int)
parser.add_argument("--attention_dim", default=128, type=int)
parser.add_argument("--rnn", default="lstm")
parser.add_argument("--trainer", default="simple_sgd")

## experiment parameters
parser.add_argument("--mmi", action='store_true')
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--learning_rate", default=1.0, type=float)
parser.add_argument("--log_train_every_n", default=500, type=int)
parser.add_argument("--log_valid_every_n", default=10000, type=int)
parser.add_argument("--output")

## choose what model to use
parser.add_argument("--model", default="basic")
parser.add_argument("--reader_mode")
parser.add_argument("--load")
parser.add_argument("--save")
parser.add_argument("--eval")

## model-specific parameters
parser.add_argument("--beam_size", default=3, type=int)
parser.add_argument("--minibatch_size", default=64, type=int)

args = parser.parse_args()
print "ARGS:", args

if args.rnn == "lstm": args.rnn = dynet.LSTMBuilder
elif args.rnn == "gru": args.rnn = dynet.GRUBuilder
else: args.rnn = dynet.SimpleRNNBuilder

BEGIN_TOKEN = '<s>'
END_TOKEN = '<e>'

# define model and obtain vocabulary
# (reload vocab files is saved model or create new vocab files if new model)

model = dynet.Model()

if not args.trainer or args.trainer=="simple_sgd":
    trainer = dynet.SimpleSGDTrainer(model)
elif args.trainer == "momentum_sgd":
    trainer = dynet.MomentumSGDTrainer(model)
elif args.trainer == "adadelta":
    trainer = dynet.AdadeltaTrainer(model)
elif args.trainer == "adagrad":
    trainer = dynet.AdagradTrainer(model)
elif args.trainer == "adam":
    trainer = dynet.AdamTrainer(model)
else:
    raise Exception("Trainer not recognized! Please use one of {simple_sgd, momentum_sgd, adadelta, adagrad, adam}")

trainer.set_clip_threshold(-1.0)
trainer.set_sparse_updates(True)

# load corpus

print "Loading corpus..."
train_data = list(util.get_reader(args.reader_mode)(args.train, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
if args.valid:
    valid_data = list(util.get_reader(args.reader_mode)(args.valid, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN))
else:
    if args.percent_valid > 1: cutoff = args.percent_valid
    else: cutoff = int(len(train_data)*(args.percent_valid))
    valid_data = train_data[-cutoff:]
    train_data = train_data[:-cutoff]
    print "Train set of size", len(train_data), "/ Validation set of size", len(valid_data)
print "done."

# Initialize model
S2SModel = seq2seq.get_s2s(args.model)
if args.load:
    print "Loading existing model..."
    s2s = S2SModel.load(model, args.load)
    src_vocab = s2s.src_vocab
    tgt_vocab = s2s.tgt_vocab
else:
    print "New model. Getting vocabulary from training set...",
    src_reader = util.get_reader(args.reader_mode)(args.train, mode=args.reader_mode, begin=BEGIN_TOKEN, end=END_TOKEN)
    src_vocab = util.Vocab.load_from_corpus(src_reader, remake=args.rebuild_vocab, src_or_tgt="src")
    src_vocab.START_TOK = src_vocab[BEGIN_TOKEN]
    src_vocab.END_TOK = src_vocab[END_TOKEN]
    src_vocab.add_unk(args.unk_thresh)
    if args.mmi:
        tgt_vocab = src_vocab
    else:
        tgt_reader = util.get_reader(args.reader_mode)(args.train, mode=args.reader_mode, end=END_TOKEN)
        tgt_vocab = util.Vocab.load_from_corpus(tgt_reader, remake=args.rebuild_vocab, src_or_tgt="tgt")
        tgt_vocab.END_TOK = tgt_vocab[END_TOKEN]
        tgt_vocab.add_unk(args.unk_thresh)
    print "Source vocabulary of size", src_vocab.size, " and target vocab of size", tgt_vocab.size
    print "Creating model..."
    s2s = S2SModel(model, src_vocab, tgt_vocab, args)
    print "...done!"


# create log file for training
if args.output:
    outfile = open(args.output, 'w')
    outfile.write("")
    outfile.close()

train_data.sort(key=lambda x: -len(x))
valid_data.sort(key=lambda x: -len(x))

# Store starting index of each minibatch
train_order = [x*args.minibatch_size for x in range(int(len(train_data)/args.minibatch_size + 1))]
valid_order = [x*args.minibatch_size for x in range(int(len(valid_data)/args.minibatch_size + 1))]

# run training loop
word_count = sent_count = cum_loss = 0.0

try:
    for ITER in range(args.epochs):
        s2s.epoch = ITER
        random.shuffle(train_order)
        sample_num = 0
        log_start = time.time()
        _start = time.time()
        for i, sid in enumerate(train_order):

            #Retrieving batch from training data
            batched_src = [tup[0] for tup in train_data[sid : sid + args.minibatch_size]]
            batched_tgt = [tup[1] for tup in train_data[sid : sid + args.minibatch_size]]
            sample_num += 1

            if sample_num % (args.log_train_every_n/args.minibatch_size) == 0:
                print("[training_set] Epoch:", ITER, "Batch:", sample_num)
                trainer.status()
                print("Loss:", cum_loss / word_count, "Time elapsed:", (time.time() - _start) ,"WPS:", word_count/(time.time() - log_start))
                sample = s2s.generate(batched_src[0], sampled=False)
                if sample: print src_vocab.pp(batched_src[0], ' '), tgt_vocab.pp(batched_tgt[0], ' '), tgt_vocab.pp(sample, ' '),
                word_count = sent_count = cum_loss = 0.0
                log_start = time.time()
                print
            # end of test logging

            if sample_num % (args.log_valid_every_n/args.minibatch_size) == 0:
                v_word_count = v_sent_count = v_cum_loss = v_cum_bleu = v_cum_em = 0.0
                v_start = time.time()
                for vid in valid_order:
                    batched_v_src = [tup[0] for tup in valid_data[vid : vid + args.minibatch_size]]
                    batched_v_tgt = [tup[1] for tup in valid_data[vid : vid + args.minibatch_size]]
                    v_loss = s2s.get_batch_loss(batched_v_src, batched_v_tgt)
                    v_cum_loss += v_loss.scalar_value()
                    # v_cum_em += s2s.get_em(batched_v_src, batched_v_tgt)
                    # v_cum_bleu += s2s.get_bleu(v_src, v_tgt, args.beam_size)
                    v_word_count += sum([(len(src_sent) - 1) for src_sent in batched_v_src])
                    v_sent_count += args.minibatch_size
                print("[Validation Set"+str(sample_num) + "]\t" + \
                      "Loss: "+str(v_cum_loss / v_word_count) + "\t" + \
                      "Perplexity: "+str(math.exp(v_cum_loss / v_word_count)) + "\t" + \
                      # "BLEU: "+str(v_cum_bleu / v_sent_count) + "\t" + \
                      # "EM: "  +str(v_cum_em   / v_sent_count) + "\t" + \
                      "Time elapsed: "+str(time.time() - v_start))
                if args.log_output:
                    print("(logging to", args.log_output + ")")
                    with open(args.log_output, "a") as outfile:
                        outfile.write(str(ITER) + "\t" + \
                                      str(sample_num) + "\t" + \
                                      str(v_cum_loss / v_word_count) + "\t" + \
                                      str(math.exp(v_cum_loss / v_word_count)) + "\t" + \
                                      # str(v_cum_em   / v_sent_count) + "\t" + \
                                      # str(v_cum_bleu / v_sent_count) + "\n")
                print
                if args.save:
                    print("saving checkpoint...")
                    s2s.save(args.save + ".checkpoint")
            # end of validation logging

            # loss = s2s.get_loss(src, tgt)
            loss = s2s.get_batch_loss(batched_src, batched_tgt)
            cum_loss += loss.value()
            word_count += sum([(len(s)-1) for s in batched_src])
            sent_count += args.minibatch_size
            loss.backward()
            trainer.update(args.learning_rate)

            ### end of one-sentence train loop
        trainer.update_epoch(args.learning_rate)
        ### end of iteration
    ### end of training loop
except KeyboardInterrupt:
    if args.save:
        print "saving..."
        s2s.save(args.save)
        sys.exit()

if args.save:
    print "saving..."
    s2s.save(args.save)

# coding: utf-8
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from apex import amp
except:
    print('Failed to import apex. You can still train with --precision {float|double}.')

from mup.coord_check import get_coord_data, plot_coord_data
from mup import MuAdam, MuSGD, get_shapes, make_base_shapes, set_base_shapes

import data
import model as mdl


###############################################################################
# Training code
###############################################################################

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def batchloader(train_data, bptt):
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        yield get_batch(train_data, i, bptt)

def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
    
def setprec(t, precision):
    if precision == 'half':
        # do nothing since this is handled by AMP
        return t
    elif precision == 'float':
        return t.float()
    elif precision == 'double':
        return t.double()
    else:
        raise ValueError(f'invalid precision string {args.precision}')

def coord_check(mup, lr, optimizer, batch_size, nsteps, nseeds, data_dir, args, plotdir='', legend=False):

    corpus = data.Corpus(data_dir)
    ntokens = len(corpus.dictionary)

    def gen(w, standparam=False):
        import model as _model
        def f():
            model = _model.TransformerModel(args, ntokens, ninp=w, nhead=args.nhead, nhid=w*args.ffn_ratio, nlayers=args.nlayers, dropout=args.dropout,
                                            tied=args.tied, bias=args.bias, encoder_var=args.init_var, 
                                            decoder_var=args.init_var, standparam=standparam).to(args.device)
            model = setprec(model, args.precision)
            if standparam:
                set_base_shapes(model, None)
            else:
                assert args.load_base_shapes, 'load_base_shapes needs to be nonempty'
                set_base_shapes(model, args.load_base_shapes)
            return model
        return f

    optimizer = optimizer.replace('mu', '')
    widths = 2**np.arange(7, 14 if optimizer=='sgd' else 12)
    models = {w: gen(w, standparam=not mup) for w in widths}

    
    train_data = batchify(corpus.train, batch_size, device=args.device)
    df = get_coord_data(models, batchloader(train_data, args.bptt), mup=mup, lr=lr, optimizer=optimizer, flatten_output=True, nseeds=nseeds, nsteps=nsteps, lossfn='nll')

    prm = 'μP' if mup else 'SP'
    return plot_coord_data(df, legend=legend,
        save_to=os.path.join(plotdir, f'{prm.lower()}_trsfmr_{optimizer}_coord.png'),
        suptitle=f'{prm} Transformer {optimizer} lr={lr} nseeds={nseeds}',
        face_color='xkcd:light grey' if not mup else None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=
    '''
    PyTorch Wikitext-2 Transformer Language Model, with μP.

    To train a μP model, one needs to first specify the base shapes. To save base shapes info, run, for example,

        python main.py --d_model 256 --save_base_shapes width256.bsh

    To train using MuAdam, run

        python main.py --d_model 256 --load_base_shapes width256.bsh --cuda --optimizer muadam

    To perform coord check, run

        python main.py --load_base_shapes width256.bsh --optimizer sgd --lr 0.5 --cuda --coord_check

        python main.py --load_base_shapes width256.bsh --optimizer adam --lr 0.01 --cuda --coord_check

    If you don't specify a base shape file, then you are using standard parametrization

        python main.py --d_model 256 --cuda --optimizer muadam

    Note that models of different depths need separate `.bsh` files.
    ''', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--bias', action='store_true',
                        help='use bias')
    parser.add_argument('--save_base_shapes', type=str, default='',
                        help='file location to save base shapes at')
    parser.add_argument('--load_base_shapes', type=str, default='',
                        help='file location to load base shapes from')
    parser.add_argument('--d_model', type=int, default=256,
                        help='width of the model')
    parser.add_argument('--ffn_ratio', type=int, default=1,
                        help='the ratio of d_ffn to d_model')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0,
                        help='momentum')
    parser.add_argument('--output_mult', type=float, default=1,
                        help='output is multiplied by sqrt(output_mult/d_model)')
    parser.add_argument('--input_mult', type=float, default=1,
                        help='input is multiplied by sqrt(input_mult*d_model)')
    parser.add_argument('--attn_mult', type=float, default=1,
                        help='attn is multiplied by sqrt(attn_mult)/head_dim')
    parser.add_argument('--optimizer', default='musgd', choices=['sgd', 'musgd', 'adam', 'muadam'])
    parser.add_argument('--init_var', type=float, default=1,
                        help='weights are initialized with variance init_var/ninp')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--precision', type=str, default='float',
                        help='float | double | half')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='path to save the final model')
    parser.add_argument('--resume_dir', type=str, default=None,
                        help='path to resume training')
    parser.add_argument('--log_dir', type=str, default='.',
                        help='path to save logs')
    parser.add_argument('--coord_check', action='store_true',
                        help='test μ parametrization is correctly implemented by collecting statistics on coordinate distributions for a few steps of training.')
    parser.add_argument('--coord_check_nsteps', type=int, default=3,
                        help='Do coord check with this many steps.')
    parser.add_argument('--coord_check_nseeds', type=int, default=3,
                        help='number of seeds for testing correctness of μ parametrization')
    parser.add_argument('--deferred_init', action='store_true', help='Skip instantiating the base and delta models for mup. Requires torchdistx.')
    
    args = parser.parse_args()

    print(args)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = args.device = torch.device("cuda" if args.cuda else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data)

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    ###############################################################################
    # Build the model
    ###############################################################################


    ntokens = len(corpus.dictionary)



    def evaluate(data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i, args.bptt)
                output = model(data)
                output = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)


    def train(optimizer, epoch):
        # Turn on training mode which enables dropout.
        model.train()
        total_loss = 0.
        epoch_loss = 0.
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        first_loss = None
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            data, targets = get_batch(train_data, i, args.bptt)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            
            optimizer.zero_grad()
            output = model(data)
            output = output.view(-1, ntokens)
            loss = criterion(output, targets)
            if torch.isnan(loss):
                exit(0)
            if args.precision == 'half':
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.clip > 0:
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                if args.precision == 'half':
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            total_loss += loss.item()
            epoch_loss += len(data) * loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // args.bptt, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, np.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
                if first_loss is None:
                    first_loss = cur_loss

        return epoch_loss / (len(train_data) - 1), first_loss
        
    if args.coord_check:
        print('testing parametrization')
        import os
        os.makedirs('coord_checks', exist_ok=True)
        plotdir = 'coord_checks'
        coord_check(mup=True, lr=args.lr, optimizer=args.optimizer, batch_size=args.batch_size, nsteps=args.coord_check_nsteps, nseeds=args.coord_check_nseeds, data_dir=args.data, args=args, plotdir=plotdir, legend=False)
        coord_check(mup=False, lr=args.lr, optimizer=args.optimizer, batch_size=args.batch_size, nsteps=args.coord_check_nsteps, nseeds=args.coord_check_nseeds, data_dir=args.data, args=args, plotdir=plotdir, legend=False)
        import sys; sys.exit()


    if args.save_base_shapes:
        print(f'saving base shapes at {args.save_base_shapes}')
        if args.deferred_init:
            from torchdistx.deferred_init import deferred_init
            # We don't need to instantiate the base and delta models
            base_shapes = get_shapes(
                deferred_init(mdl.TransformerModel, args, ntokens, ninp=args.d_model, nhead=args.nhead, nhid=args.d_model*args.ffn_ratio, nlayers=args.nlayers, dropout=args.dropout,
                                        tied=args.tied, bias=args.bias, encoder_var=args.init_var, 
                                        decoder_var=args.init_var, standparam=args.load_base_shapes=='')
            )
            delta_shapes = get_shapes(
                # just need to change whatever dimension(s) we are scaling
                deferred_init(mdl.TransformerModel, args, ntokens, ninp=args.d_model*2, nhead=args.nhead, nhid=args.d_model*args.ffn_ratio*2,
                                        nlayers=args.nlayers, dropout=args.dropout,
                                        tied=args.tied, bias=args.bias, encoder_var=args.init_var, 
                                        decoder_var=args.init_var, standparam=args.load_base_shapes=='')
            )
        else:
            base_shapes = get_shapes(
                mdl.TransformerModel(args, ntokens, ninp=args.d_model, nhead=args.nhead, nhid=args.d_model*args.ffn_ratio, nlayers=args.nlayers, dropout=args.dropout,
                                        tied=args.tied, bias=args.bias, encoder_var=args.init_var, 
                                        decoder_var=args.init_var, standparam=args.load_base_shapes=='')
            )
            delta_shapes = get_shapes(
                # just need to change whatever dimension(s) we are scaling
                mdl.TransformerModel(args, ntokens, ninp=args.d_model*2, nhead=args.nhead, nhid=args.d_model*args.ffn_ratio*2,
                                        nlayers=args.nlayers, dropout=args.dropout,
                                        tied=args.tied, bias=args.bias, encoder_var=args.init_var, 
                                        decoder_var=args.init_var, standparam=args.load_base_shapes=='')
            )
        make_base_shapes(base_shapes, delta_shapes, savefile=args.save_base_shapes)
        print('done and exit')
        import sys; sys.exit()
    model = mdl.TransformerModel(args, ntokens, ninp=args.d_model, nhead=args.nhead, nhid=args.d_model*args.ffn_ratio, nlayers=args.nlayers, dropout=args.dropout,
                                    tied=args.tied, bias=args.bias, encoder_var=args.init_var, 
                                    decoder_var=args.init_var, standparam=args.load_base_shapes=='')
    if args.load_base_shapes:
        print(f'loading base shapes from {args.load_base_shapes}')
        set_base_shapes(model, args.load_base_shapes)
        print('done')
    else:
        print(f'using own shapes')
        set_base_shapes(model, None)
        print('done')

    model = model.to(device)
    model = setprec(model, args.precision)

    criterion = nn.NLLLoss()

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    # Loop over epochs.
    lr = args.lr
    best_val_loss = float('inf')

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'musgd':
        optimizer = MuSGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'muadam':
        optimizer = MuAdam(model.parameters(), lr=args.lr)
    else:
        raise ValueError()

    # half-precision black magic
    if args.precision == 'half':
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level='O1',
            min_loss_scale=0.0001,
            verbosity=0
            )

    logs = []
    start_epoch = 0
    if args.resume_dir and os.path.exists(os.path.join(args.resume_dir, 'checkpoint_last.pt')):
        checkpoint = torch.load(os.path.join(args.resume_dir, 'checkpoint_last.pt'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.precision == 'half':
            amp.load_state_dict(checkpoint['amp'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        logs = checkpoint['logs']

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(start_epoch+1, args.epochs+1):
            epoch_start_time = time.time()
            train_loss, first_loss = train(optimizer, epoch)
            # print(first_loss)
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, np.exp(val_loss)))
            print('-' * 89)
            logs.append(dict(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                first_loss=first_loss
            ))
            # Save the model if the validation loss is the best we've seen so far.
            if args.save_dir is not None:
                if val_loss < best_val_loss:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_val_loss': best_val_loss,
                        'logs': logs
                    }
                    if args.precision == 'half':
                        checkpoint['amp'] = amp.state_dict(),
                    with open(os.path.join(args.save_dir, 'checkpoint_best.pt'), 'wb') as f:
                        torch.save(checkpoint, f)
                    best_val_loss = val_loss
                else:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_val_loss': best_val_loss,
                        'logs': logs
                    }
                    if args.precision == 'half':
                        checkpoint['amp'] = amp.state_dict()
                with open(os.path.join(args.save_dir, 'checkpoint_last.pt'), 'wb') as f:
                    torch.save(checkpoint, f)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    if args.save_dir is not None:
        with open(os.path.join(args.save_dir, 'checkpoint_best.pt'), 'rb') as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.precision == 'half':
                amp.load_state_dict(checkpoint['amp'][0])
        # Run on test data.
        test_loss = evaluate(test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, np.exp(test_loss)))
        print('=' * 89)
        logs.append(dict(
            epoch='-1',
            test_loss=test_loss
        ))


    with open(os.path.join(os.path.expanduser(args.log_dir), 'logs.tsv'), 'w') as f:
        logdf = pd.DataFrame(logs)
        print(os.path.join(os.path.expanduser(args.log_dir), 'logs.tsv'))
        f.write(logdf.to_csv(sep='\t', float_format='%.4f'))

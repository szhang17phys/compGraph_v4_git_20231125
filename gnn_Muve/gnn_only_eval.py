import time
from optparse import OptionParser
from utlis_only_eval import *
#from utlis_only_eval_checkSTD import *

usage = 'usage: %prog [-i infile] [-o outfile] [-e epochs] [-b batchsize] [-n nfile] [-t tier]'

parser = OptionParser(usage=usage)
parser.add_option('-i',  '--infile',    dest='infile',    help='Input location',   default='/wclustre/dune/muve/gan/dataset/')
parser.add_option('-o',  '--outfile',   dest='outfile',   help='Output location',  default='./')
parser.add_option('-b',  '--batchsize', dest='batchsize', help='batch size',       type='int', default=4096)
parser.add_option('-e',  '--epochs',    dest='epochs',    help='epochs to train',  type='int', default=10000)
parser.add_option('-n',  '--nfile',     dest='nfile',     help='files to load',    type='int', default=10000)
parser.add_option('-t',  '--tier',      dest='tier',      help='modle to load',    type='int', default=0)
parser.add_option('-d',  '--dim',       dest='dim',       help='dim of pdr',       type='int', default=90)
parser.add_option('-p',  '--opt',       dest='opt',       help='optimizer',        default='Adam')
parser.add_option('--train',            dest='train',     help='train network',    default=False,  action='store_true')
parser.add_option('--eval',             dest='eval',      help='evaluate network', default=False,  action='store_true')
parser.add_option('--graph',            dest='graph',     help='freeze model',     default=False,  action='store_true')
parser.add_option('--debug',            dest='debug',     help='debug code',       default=False,  action='store_true')

if __name__ == '__main__':

    (options, args) = parser.parse_args()
    
    if len(args) > 0:
        print('Cannot handle unspecified arguments', args)
        sys.exit(1)
        
    inpath    = options.infile
    mtier     = options.tier
    modpath   = options.outfile+str(mtier)+'-mod/'
    evlpath   = options.outfile+str(mtier)+'-evl/'
    nfile     = options.nfile
    epochs    = options.epochs
    batchsize = options.batchsize
    opt       = options.opt
    dim_pdr   = options.dim
    dim_pos   = 3
    
    if options.train:
        mkdir(modpath)
        pos, pdr = get_data(inpath, nfile, dim_pos, dim_pdr)
        tstart = time.time()
        print( 'Train the network with epoch: '+str(epochs)+' batch: '+str(batchsize)+'.')
        train(pos, pdr, mtier, epochs, batchsize, modpath, opt)        
        print( 'Finish training in '+str(time.time()-tstart)+'s.')
        
    if options.eval:
        mkdir(evlpath)
        pos, pdr = get_data(inpath, nfile, dim_pos, dim_pdr)
        tstart = time.time()
        print( 'Evaluating network performance...')
        eval(pos, pdr, mtier, modpath, evlpath)
        print( 'Finish evaluation in '+str(time.time()-tstart)+'s.')
        
    if options.graph:
        freezemodel(modpath)

    if options.debug:
        debug(dim_pdr, mtier, opt)

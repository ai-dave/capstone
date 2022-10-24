import itertools
import zipfile
import re
import json


def read_fasta(dir_file, limit=0, lowercase=False):
    import sys, re

    f=open(dir_file,'r')
    lines=f.readlines()

    hre=re.compile('>(\S+)')
    lre=re.compile('^(\S+)$')

    gene={}
    i=0
    
    for line in lines:
        outh = hre.search(line)
        if outh:
            id=outh.group(1)
        else:
            if lowercase == True:
                line = line.lower()
            outl=lre.search(line)
            if(id in gene.keys()):
                gene[id] += outl.group(1)
            else:
                gene[id]  =outl.group(1)
        i += 1
        if limit>0 and i>=limit:
            break
    return gene

def read_fasta_ncbi_dataset_zip(ncbi_dataset_zipfile, file, limit=0, lowercase=False):
    import sys, re

    with zipfile.ZipFile(ncbi_dataset_zipfile) as myzip:
        myzip.extract(file)

    f=open(file,'r')
    lines=f.readlines()

    hre=re.compile('>(\S+)')
    lre=re.compile('^(\S+)$')

    gene={}
    i=0

    for line in lines:
        outh = hre.search(line)
        if outh:
            id=outh.group(1)
        else:
            outl=lre.search(line)
            if(id in gene.keys()):
                gene[id] += outl.group(1)
            else:
                gene[id]  =outl.group(1)
        i += 1
        if limit>0 and i>=limit:
            break
    return gene


def perm(n, seq):
    # Actually returns cross-product of alphabet
    l=[]
    for p in itertools.product(seq, repeat=n):
        l.append(''.join(p))
        #l.append(p)
    return l


def make_dictionary(n, alphabet):
    words = perm(n, alphabet)
    d = dict(zip(words, range(len(words))))
    return d


def make_kmer(k, alphabet, seq):
    dictionary = make_dictionary(k, alphabet)
    kmer = [dictionary[seq[x:x+k]] for x in range(len(seq)-(k-1))]
    return kmer


def make_kmer_sentence(k, seq):
    kmer = ''
    for x in range(len(seq)-(k-1)):
        kmer += seq[x:x+k]+' '
    return kmer


def make_kmer_sentence_list(k, seq):
    kmer = make_kmer_sentence(k, seq)
    return kmer.split(' ')


def make_ncbi_dataset(ncbi_dataset_zipfile):
    with zipfile.ZipFile(ncbi_dataset_zipfile) as myzip:
        dataset = myzip.namelist()
        #print(*(x for x in d), sep='\n')

    p = re.compile('^ncbi_dataset/data/(\S+)/(\S+\.(\S+))$')
    ncbi_dataset = {}
    for x in dataset:
        if (x.find('.fna') != -1) or (x.find('.gff') != -1) or (x.find('.faa') != -1):
            #print(f'[{x}]')
            m = p.match(x)
            accession_number = m.group(1)
            filename         = m.group(2)
            extension        = m.group(3)
            extension        = "fna2" if (extension == 'fna' and filename != 'cds_from_genomic.fna') else m.group(3)

            #print(f'[{accession_number}] - [{filename}] - [{extension}]')
            if accession_number not in ncbi_dataset:
                ncbi_dataset[accession_number] = {extension : filename}
            else:
                ncbi_dataset[accession_number].update({extension : filename})
    return ncbi_dataset


def files_ncbi_dataset(ncbi_dataset_zipfile):
    with zipfile.ZipFile(ncbi_dataset_zipfile) as myzip:
        return myzip.namelist()


def make_assembly_data_report(ncbi_dataset_zipfile):
    report_file = 'ncbi_dataset/data/assembly_data_report.jsonl'
    with zipfile.ZipFile(ncbi_dataset_zipfile) as myzip:
        myzip.extract(report_file)

    assembly_data_report = [json.loads(line) for line in open(report_file, 'r', encoding='utf-8')]
    return assembly_data_report



def make_sequence_report(ncbi_dataset_zipfile, assemblyAccession):
    sequence_report_file = "ncbi_dataset/data/" + assemblyAccession + "/sequence_report.jsonl"
    #print(sequence_report_file)

    with zipfile.ZipFile(ncbi_dataset_zipfile) as myzip:
        myzip.extract(sequence_report_file)

    sequence_report = [json.loads(line) for line in open(sequence_report_file, 'r', encoding='utf-8')]
    return sequence_report

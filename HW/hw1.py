%%writefile nb_hw1_2.py
from mrjob.job import MRJob
import re

Corpus = dict()
class WCMrJob(MRJob):
    
    def __init__(self, *args, **kwargs):
        super(WCMrJob, self).__init__(*args, **kwargs)
        self.Corpus = dict()

    def mapper(self, _ , line):
        
        words = line.split('\t')
        #print('words:')
        #print(words)
        for j in words[2:]:
            cleantxt = re.sub('\W+',' ',j)
            splitclean = cleantxt.split()
            #print('cleantxt:')
            #print(cleantxt)
            #print('splitclean:')
            #print(splitclean)
            for i in splitclean:
                yield(i , 1)
                #print(i, 1)
            
    def reducer(self, key , values):
        yield (key , sum(values))
        self.Corpus[key] = sum(values)
        Corpus[key] = sum(values)
        
        #print('self.Corpus:')
        #yield('self.Corpus:' , self.Corpus)

#print(Corpus['assistance'])

if __name__ == '__main__':
    WCMrJob.run() 

#coding: utf8
#author: Tian Xia (SummerRainET2008@gmail.com)

import nltk
import bleu_measure

if __name__ == "__main__":
  ref1 = "a b c d 1 2 3 4 5 6 7 8 e"
  ref2 = "a b c d 1 2 3 4 5 6 7 8 e"
  ref3 = "a b c e 1 2 3 4 5 6 7 8 d"
  hyp  = "a b c e 1 2 3 4 5 6 7 e e"

  bleu = nltk.translate.bleu_score.sentence_bleu([ref1.split(),
                                                  ref2.split(),
                                                  ref3.split()],
                                                 hyp.split(),
                                                 [1, 1, 1])
  print(f"nltk BLEU: {bleu}")

  #todo
  sent_bleu = bleu_measure.SentenceBLEU([ref1, ref2, ref3], 3)
  print(f"BLEU1:", sent_bleu.compute(hyp))

  relsList = []
  relsList.append([3, 2, 1] + [0] * 20 + [4] * 2)
  print("relsList:", relsList)
  print("ndcg:", bleu_measure.calc_ndcg(relsList))


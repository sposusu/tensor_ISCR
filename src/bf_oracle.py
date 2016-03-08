from multiprocessing import Pool
seqs = []
def get_seqs():
  seqs = []
  # 1
  for a in xrange(4):
    seqs.append([a])

  # 2
  for a in xrange(4):
    for b in xrange(4):
      seqs.append([a,b])

  # 3
  for a in xrange(4):
    for b in xrange(4):
      for c in xrange(4):
        seqs.append([a,b,c])

  # 4
  for a in xrange(4):
    for b in xrange(4):
      for c in xrange(4):
        for d in xrange(4):
          seqs.append([a,b,c,d])
  return seqs

def test_action():
  pool = Pool()
  result1 = pool.apply_async(solve1, [A])    # evaluate "solve1(A)" asynchronously
  result2 = pool.apply_async(solve2, [B])    # evaluate "solve2(B)" asynchronously

  env = setEnvironment()
  best_returns = - np.ones(163)
  best_seqs = defaultdict(list)
  APs = np.zeros(163)
  seqs = get_seqs()

  for idx in xrange(len(data)):
    print '\nQuery ',idx
    best_returns[idx], best_seqs[idx], APs[idx] = test_one_action(idx)

  filename = 'result/' + '.'.join(rec_type) + '_best_seq_return.pkl'
  with open(filename,'w') as f:
    pickle.dump( (best_returns, best_seqs,APs),f )
  print 'MAP = ', np.mean(APs),'Return = ',np.mean(Returns)

def test_one_action(idx):
  print '\nQuery ',idx
    for seq in seqs:
      cur_return = 0.
      init_state = env.setSession(q,ans,ans_index,True)
      for act in seq:
        reward, state = env.step(act)
        cur_return += reward
      terminal, AP = env.game_over()
      sys.stderr.write('\rActions Sequence {}    Return = {}'.format(seq,cur_return))

      if cur_return > best_returns[idx]:
        best_returns[idx] = cur_return
        best_seqs[idx] = seq
        APs[idx] = AP
    print '\rBest seq :', best_seqs[idx],'    Best Return : ', best_returns[idx],'    AP : ', APs[idx]
  return


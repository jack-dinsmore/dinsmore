from multiprocessing import Pool as Multipool

class Pool:
    def __init__(self, n_threads):
        if n_threads == 1:
            self.pool = None
        else:
            self.pool = Multipool(n_threads)
    
    def close(self):
        if self.pool is not None:
            self.pool.close()

    def map(self, func, args):
        if len(args) == 0: return []
        if self.pool is None:
            results = []
            for arg in args:
                results.append(func(arg))
        else:
            results = self.pool.map(func, args)
        return results
    
    def starmap(self, func, args):
        if len(args) == 0: return []
        if self.pool is None:
            results = []
            for arg in args:
                results.append(func(*arg))
        else:
            results = self.pool.starmap(func, args)
        return results
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def __enter__(self):
        return self
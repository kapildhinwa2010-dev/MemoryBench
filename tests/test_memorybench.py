import pytest
import torch
from memorybench import MemoryBench, create_memorybench

class TestMemoryBench:
    def test_init(self):
        m = MemoryBench(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = MemoryBench(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_memorybench()
        assert isinstance(m, MemoryBench)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

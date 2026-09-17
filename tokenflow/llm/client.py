from tokenflow.core.optimizer import Optimizer
from tokenflow.llm.providers import Provider, ProviderResult
from tokenflow.models import OptimizationRequest, OptimizationResult, StrictModel


class GenerationResult(StrictModel):
    response: ProviderResult
    optimization: OptimizationResult


class TokenFlow:
    def __init__(self, provider: Provider, optimizer: Optimizer | None = None):
        self.provider = provider
        self.optimizer = optimizer or Optimizer()

    def generate(self, request: OptimizationRequest) -> GenerationResult:
        optimization = self.optimizer.optimize(request)
        response = self.provider.generate(optimization.messages)
        return GenerationResult(response=response, optimization=optimization)

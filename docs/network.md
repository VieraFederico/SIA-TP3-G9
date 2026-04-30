# network/ — La Red Neuronal

Tres clases con relación de composición directa.

## Clases

### `Neuron` (`neuron.py`)
Una sola neurona manual. Recibe N entradas y devuelve una salida.

**Qué hace:**
1. Calcula la excitación: `h = Σ xᵢwᵢ + w₀`
2. Aplica la función de activación: `O = θ(h)`
3. Guarda `h` para usarlo en backpropagation

**Nota:** `Neuron` es la implementación de referencia (no vectorizada). En el pipeline real
se usa siempre `NeuronLayer`, que es equivalente pero eficiente con NumPy.

### `NeuronLayer` (`neuron_layer.py`)
Capa de N neuronas vectorizada con NumPy. Equivale a una fila de `Neuron`s.

**Constructor:** `NeuronLayer(n_inputs, n_neurons, activation)`

**Inicialización de pesos:**
```
W  ~ N(0, 0.01)   shape (n_inputs, n_neurons)
b  = zeros         shape (n_neurons,)
```

**Qué hace:**
- `forward(x)`: calcula `h = x·W + b`, luego `V = θ(h)`. Guarda `_x`, `_h` y `_V`.
- `backward(delta)`: recibe `δ = ∂E/∂V` de la capa siguiente.
  - Si `activation.is_differentiable()`: `δh = δ · θ'(h)` — regla de la cadena
  - Si no (Step): `δh = δ` — regla de Rosenblatt (trata θ'(h) = 1)
  - Calcula `grad_weights = outer(_x, δh)` y `grad_bias = δh`
  - Devuelve `W @ δh` (gradiente para la capa anterior)
- `get_weights() → (W, b)`
- `set_weights(W, b)`

**Por qué guarda `_h` y `_V`:**
Durante el backward se necesita `θ'(h)` para calcular `δh`.

### `MultilayerPerceptron` (`multilayer_perceptron.py`)
Lista de `NeuronLayer`. Orquesta el forward y backpropagation completos.

**Constructor:** `MultilayerPerceptron(layers: list[NeuronLayer])`

**Un perceptrón simple es un caso especial:**
`MultilayerPerceptron` con una sola `NeuronLayer` de una neurona.

**Forward pass:** `x → V¹ → V² → ... → O`
```python
def forward(self, x):
    for layer in self.layers:
        x = layer.forward(x)
    return x
```

**Backpropagation:**
```python
def backward(self, grad_output):
    delta = grad_output
    for layer in reversed(self.layers):
        delta = layer.backward(delta)
```

**Acceso a pesos:**
- `get_weights() → [(W⁰, b⁰), (W¹, b¹), ...]` — una tupla por capa
- `get_grads()  → [(gW⁰, gb⁰), (gW¹, gb¹), ...]` — gradientes acumulados
- `set_weights([(W⁰, b⁰), ...])` — asigna todos los pesos de una vez

## Relación con la teoría

```
Clase                  Símbolo teoría
──────────────────────────────────────
Neuron.h               h (excitación)
Neuron.O               O (salida neurona)
NeuronLayer._V         V (salida de la capa)
NeuronLayer.weights    W (pesos de la capa)
NeuronLayer.bias       b (bias)
delta_h                δ = ∂E/∂h = ∂E/∂V · θ'(h)
grad_weights           ∂E/∂W = xᵀ · δh
```

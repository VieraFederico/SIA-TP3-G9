# network/ — La Red Neuronal

Tres clases con relación de composición directa.

## Clases

### `Neuron` (`neuron.py`)
Una sola neurona. Recibe N entradas y devuelve una salida.

**Qué hace:**
1. Calcula la excitación: `h = Σ xᵢwᵢ + w₀`
2. Aplica la función de activación: `O = θ(h)`
3. Guarda `h` para usarlo en backpropagation

**Parámetros del constructor:**
- `n_inputs`: número de entradas
- `activation`: instancia de `ActivationFunction`

### `NeuronLayer` (`neuron_layer.py`)
Una capa de N neuronas, implementada de forma vectorizada con NumPy.
Equivale a una fila de `Neuron`s pero eficiente.

**Qué hace:**
- `forward(x)`: calcula `h = X·W + b`, luego `V = θ(h)`. Guarda `h` y `V`.
- `backward(delta)`: recibe `δ` de la capa siguiente, calcula `delta_w` y `δ` para la capa anterior.

**Por qué guarda h y V:**
Durante el backward necesitamos `θ'(h)` para calcular `δ = δ_siguiente · θ'(h)`.

### `MultilayerPerceptron` (`multilayer_perceptron.py`)
Lista de `NeuronLayer`. Orquesta el forward y backpropagation completos.

**Un perceptrón simple es un caso especial:**
`MultilayerPerceptron` con una sola `NeuronLayer` de una neurona.
La activación (Step, Identity, Tanh...) se inyecta al construirlo.

**Forward pass:** `x → V¹ → V² → ... → O`

**Backpropagation:** propaga `δ` capa por capa de derecha a izquierda
usando la regla de la cadena: `∂E/∂w = δ · V_anterior`

## Relación con la teoría

```
Clase             Símbolo teoría
─────────────────────────────────
Neuron.h          h (excitación)
Neuron.O          O (salida)
NeuronLayer.V     V (salida capa oculta)
NeuronLayer.W     W (pesos de la capa)
delta             δ = (ζ - O)·θ'(h)
```

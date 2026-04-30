# Flujo completo de entrenamiento

## La idea central

Todo modelo en este proyecto — perceptrón simple, ADALINE, tanh, MLP profundo — es un
`MultilayerPerceptron` con una lista de `NeuronLayer`.  
La diferencia entre un perceptrón simple y uno multicapa es **cuántas capas tiene la lista**.

```
Perceptrón simple  →  MultilayerPerceptron( [ NeuronLayer ] )
MLP               →  MultilayerPerceptron( [ NeuronLayer, NeuronLayer, ... ] )
```

El `Trainer`, el `Optimizer` y el `MSECost` son **exactamente los mismos** en los dos casos.
La arquitectura no cambia el loop; cambia lo que pasa dentro de `forward` y `backward`.

---

## Las cuatro arquitecturas de los tutoriales

| Tutorial | Configuración | Arquitectura |
|---|---|---|
| Step (AND) | 2 entradas → 1 neurona escalón | `[NeuronLayer(2, 1, StepActivation())]` |
| Linear (ADALINE) | 1 entrada → 1 neurona identidad | `[NeuronLayer(1, 1, IdentityActivation())]` |
| No lineal (tanh) | 1 entrada → 1 neurona tanh | `[NeuronLayer(1, 1, TanhActivation(β=0.4))]` |
| MLP tanh | 1 entrada → 5 ocultas → 1 salida | `[NeuronLayer(1, 5, TanhActivation()), NeuronLayer(5, 1, TanhActivation())]` |

---

## El loop del Trainer

El `Trainer.fit` corre esto por cada época (modo `online`):

```
_train_epoch_online
    └── por cada muestra (xi, zi):
            forward      → O    = model.forward(xi)
            grad         → ∂E/∂O = cost.gradient(zi, O)
            backward     → model.backward(∂E/∂O)
            update       → model.set_weights(optimizer.update(pesos, grads))
            loss        += cost.compute(zi, O)       ← se acumula DESPUÉS del update

_evaluate_loss  ← itera todas las muestras sin backward ni update
                  devuelve la suma de cost.compute(zi, O) sobre todo val
```

El valor en `history["train_error"]` es la **suma** de pérdidas por muestra en la época,
no un promedio. El `Trainer` nunca sabe si el modelo tiene 1 capa o 10.

---

## Paso a paso por muestra — Perceptrón simple (1 capa, 1 neurona)

Ejemplo: **Perceptrón Step para AND**  
`MultilayerPerceptron([NeuronLayer(n_inputs=2, n_neurons=1, activation=StepActivation())])`

Entrada: `xi = [x₁, x₂]`  (shape `(2,)`)  
Target:  `zi = 0 ó 1`      (scalar)

### 1. Forward

```
NeuronLayer.forward(xi):
    h = xi @ W + b           →  shape (1,)   — una sola neurona, un solo h
    O = θ(h) = step(h)       →  shape (1,)   — 0.0 ó 1.0
```

`MultilayerPerceptron.forward` encadena las capas: con una sola capa, devuelve directamente el `O` de esa capa.

La capa guarda `_x = xi`, `_h = h`, `_V = O` para usarlos en el backward.

### 2. Loss y gradiente inicial

```
MSECost.compute(zi, O):
    N = len(zi) if hasattr(zi, '__len__') else 1
    E = (1/2N) · Σ (zi - O)²        →  scalar   (para 1 salida: N=1)

MSECost.gradient(zi, O):
    ∂E/∂O = -(zi - O) / N           →  shape (1,)
```

Este `∂E/∂O` es el único punto donde `zi` (el target) entra al cálculo.
A partir de acá solo circulan gradientes.

### 3. Backward

```
MultilayerPerceptron.backward(∂E/∂O):
    recorre capas en orden inverso → con 1 capa, llama a esa capa directamente

NeuronLayer.backward(δ = ∂E/∂O):
    Step no es diferenciable → δh = δ          (Rosenblatt: trata θ'(h) = 1)
    Tanh/Identity sí lo son  → δh = δ · θ'(h)

    grad_W = outer(_x, δh)         →  shape (2, 1)  — gradiente de los pesos
    grad_b = δh                    →  shape (1,)    — gradiente del bias
    return W @ δh                  →  shape (2,)    — gradiente para la capa anterior
                                                       (no se usa con 1 sola capa)
```

### 4. Update de pesos

```
Trainer._update_weights:
    params  = modelo.get_weights()   →  [(W, b)]          — lista de un par
    grads   = modelo.get_grads()     →  [(grad_W, grad_b)] — lista de un par

GradientDescent.update(params, grads):
    W_nuevo = W - learning_rate · grad_W
    b_nuevo = b - learning_rate · grad_b
    return [(W_nuevo, b_nuevo)]

modelo.set_weights([(W_nuevo, b_nuevo)])
    → NeuronLayer.set_weights(W_nuevo, b_nuevo)
```

---

## Paso a paso por muestra — MLP multicapa ([1 → 5 → 1], tutorial 4)

`MultilayerPerceptron([NeuronLayer(1, 5, TanhActivation()), NeuronLayer(5, 1, TanhActivation())])`

Entrada: `xi = [x]`  (shape `(1,)`)  
Target:  `zi = tanh(x)`  (scalar)

### 1. Forward — de izquierda a derecha

```
Capa 0 — NeuronLayer(1, 5, tanh):
    h⁰ = xi @ W⁰ + b⁰       →  shape (5,)   — 5 excitaciones, una por neurona
    V⁰ = tanh(h⁰)            →  shape (5,)   — 5 activaciones
    guarda: _x=xi, _h=h⁰, _V=V⁰

Capa 1 — NeuronLayer(5, 1, tanh):
    h¹ = V⁰ @ W¹ + b¹       →  shape (1,)   — 1 excitación (neurona de salida)
    O  = tanh(h¹)            →  shape (1,)   — predicción final
    guarda: _x=V⁰, _h=h¹, _V=O
```

`MultilayerPerceptron.forward` pasa la salida de cada capa como entrada a la siguiente.

### 2. Loss y gradiente inicial

```
MSECost.compute(zi, O):
    E = (1/2) · (zi - O)²           →  scalar   (zi es scalar → N=1)

MSECost.gradient(zi, O):
    ∂E/∂O = -(zi - O)               →  shape (1,)   (N=1)
```

### 3. Backward — de derecha a izquierda (regla de la cadena)

```
MultilayerPerceptron.backward(∂E/∂O):
    δ = ∂E/∂O                       →  shape (1,)


  ── Capa 1 (la de salida) ──────────────────────────────────────
  NeuronLayer.backward(δ = ∂E/∂O):
      δh¹ = δ · tanh'(h¹)           →  shape (1,)
      grad_W¹ = outer(V⁰, δh¹)      →  shape (5, 1)
      grad_b¹ = δh¹                 →  shape (1,)
      return W¹ @ δh¹               →  shape (5,)   ← ∂E/∂V⁰: gradiente para la capa anterior


  ── Capa 0 (la oculta) ─────────────────────────────────────────
  NeuronLayer.backward(δ = ∂E/∂V⁰):   ← recibe lo que devolvió la capa 1
      δh⁰ = δ · tanh'(h⁰)           →  shape (5,)
      grad_W⁰ = outer(xi, δh⁰)      →  shape (1, 5)
      grad_b⁰ = δh⁰                 →  shape (5,)
      return W⁰ @ δh⁰               →  shape (1,)   ← ∂E/∂xi (no se usa, ya llegamos al input)
```

Cada capa le pasa a la anterior exactamente lo que necesita para continuar la cadena.
La capa no sabe cuántas capas hay antes o después de ella.

### 4. Update de pesos — ahora hay dos capas

```
Trainer._update_weights:
    params = modelo.get_weights()    →  [(W⁰, b⁰), (W¹, b¹)]
    grads  = modelo.get_grads()      →  [(grad_W⁰, grad_b⁰), (grad_W¹, grad_b¹)]

GradientDescent.update(params, grads):
    para cada par (W, b), (gW, gb):
        W_nuevo = W - η · gW
        b_nuevo = b - η · gb
    return [(W⁰_nuevo, b⁰_nuevo), (W¹_nuevo, b¹_nuevo)]

modelo.set_weights([...])
    → capa 0 recibe (W⁰_nuevo, b⁰_nuevo)
    → capa 1 recibe (W¹_nuevo, b¹_nuevo)
```

---

## Diferencias clave entre los cuatro tutoriales

### Step vs. Tanh/Identity en el backward

`StepActivation` no es diferenciable. `NeuronLayer.backward` detecta esto con `is_differentiable()`:

```
Tanh / Identity:   δh = δ · θ'(h)    ← derivada real de la activación
Step:              δh = δ             ← regla de Rosenblatt: trata θ'(h) = 1
```

Sin este chequeo, `step'(h) = 0` en todos lados y los gradientes serían cero: los pesos nunca se moverían.

### 1 neurona vs. N neuronas

Con `n_neurons=1` todos los shapes intermedios tienen una sola dimensión de tamaño 1:
`h` es `(1,)`, `O` es `(1,)`, `grad_W` es `(n_inputs, 1)`.  
Con `n_neurons=5` (capa oculta) son `(5,)` y `(n_inputs, 5)`.  
El código de `NeuronLayer` es idéntico en los dos casos — las operaciones matriciales escalan solas.

### 1 capa vs. 2 capas en el Trainer

El `Trainer` no cambia. La diferencia es que:
- Con 1 capa: `get_weights()` devuelve una lista de un par, el optimizer actualiza un par.
- Con 2 capas: `get_weights()` devuelve una lista de dos pares, el optimizer actualiza dos pares.

---

## Resumen del flujo completo

```
Por cada época:
  Por cada muestra (xi, zi):

    [xi]
      │
      ▼  NeuronLayer.forward × M capas
    [O]   shape (n_salida,)
      │
      ▼  MSECost.compute
    [E]   scalar — solo para registrar el error
      │
      ▼  MSECost.gradient
    [∂E/∂O]   shape (n_salida,)
      │
      ▼  NeuronLayer.backward × M capas (orden inverso)
    [grad_W, grad_b por capa]
      │
      ▼  GradientDescent.update
    [W_nuevo, b_nuevo por capa]
      │
      ▼  modelo.set_weights
    pesos actualizados ✓
```

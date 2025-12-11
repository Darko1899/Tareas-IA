import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ==================== PREPARACIÓN DE DATOS ====================

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# ==================== CONSTRUCCIÓN DEL MODELO ====================

# Opción 1: Modelo simple (comentado)
# oculta = tf.keras.layers.Dense(units=1, input_shape=[1])
# modelo = tf.keras.Sequential([oculta])

# Opción 2: Modelo con capas ocultas
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[1]),  # Forma correcta de especificar input
    tf.keras.layers.Dense(units=3, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

# ==================== COMPILACIÓN ====================

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")

# ==================== VISUALIZACIÓN DEL ENTRENAMIENTO ====================

plt.figure(figsize=(10, 6))
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.title("Histórico de pérdida durante el entrenamiento")
plt.grid(True, alpha=0.3)
plt.savefig('entrenamiento.png', dpi=100)
plt.show()

# ==================== PREDICCIÓN ====================

print("\nHagamos una predicción!")
# IMPORTANTE: Convertir a numpy array con reshape
resultado = modelo.predict(np.array([100.0]).reshape(1, 1), verbose=0)
print(f"100 grados Celsius son {resultado[0][0]:.2f} Fahrenheit")

# Más predicciones
print("\nOtras predicciones:")
for celsius_valor in [0, 25, 50]:
    pred = modelo.predict(np.array([celsius_valor]).reshape(1, 1), verbose=0)
    print(f"{celsius_valor}°C = {pred[0][0]:.2f}°F")

# ==================== INFORMACIÓN DEL MODELO ====================

print("\n" + "="*60)
print("RESUMEN DEL MODELO")
print("="*60)
modelo.summary()

print("\nVariables internas del modelo (pesos):")
for i, layer in enumerate(modelo.layers):
    print(f"\nCapa {i + 1} ({layer.name}):")
    weights = layer.get_weights()
    if len(weights) > 0:
        print(f"  Pesos: {weights[0].shape}")
        print(f"  Bias: {weights[1].shape}")

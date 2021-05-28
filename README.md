``` mermaid
graph LR
x1((x1))--*w11-->b1((+b1))
x1--*w12-->b2((+b2))
x2((x1))--*w21-->b1
x2--*w22-->b2

b1--Relu-->h1((h1))
b2--Relu-->h2((h2))

h1-->softmax((softmax))
h2-->softmax

softmax-->p1((p1))
softmax-->p2((p2))
p1 --> y_pred((y_pred))
p2 --> y_pred

y_pred-->cross_entropy((cross_entropy))
y_true-->cross_entropy

cross_entropy --> loss
```

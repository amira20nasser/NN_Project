import numpy as np


class BackPropagation:
    def __init__(self, learning_rate, layers, epochs, neurons, isBias=False, isSigmoid=True):
        self.learning_rate = learning_rate
        self.layers = layers
        self.epochs = epochs
        self.isBias = isBias
        self.isSigmoid = isSigmoid
        self.neurons = neurons  # عدد النيرونات في كل طبقة
        self.z = []
        self.output = []

    def get_sizes(self, X, Y):
        """ تحديد أحجام الطبقات بناءً على المدخلات والمخرجات """
        self.input_size = X.shape[1]
        self.output_size = Y.shape[1]
        self.hidden_size = self.neurons

    def initialize_params(self, X, Y):
        """ تهيئة الأوزان والانحيازات """
        self.weights = [0.01 * np.random.randn(self.hidden_size[0], self.input_size)]
        self.bias = [0.01 * np.random.randn(1, self.hidden_size[0])] if self.isBias else [0]

        for i in range(1, self.layers):
            self.weights.append(0.01 * np.random.randn(self.hidden_size[i], self.hidden_size[i - 1]))
            if self.isBias:
                self.bias.append(0.01 * np.random.randn(1, self.hidden_size[i]))

        self.weights_output = 0.01 * np.random.randn(self.output_size, self.hidden_size[-1])
        self.bias_output = 0.01 * np.random.randn(1, self.output_size) if self.isBias else 0

    def sigmoid(self, Z):
        """ تفعيل باستخدام دالة السيجمويد """
        return 1 / (1 + np.exp(-Z))

    def tanh(self, Z):
        """ تفعيل باستخدام دالة تانجنت هايبر """
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        """ المشتقة الأولى لدالة السيجمويد """
        sigmoid_Z = self.sigmoid(Z)
        return sigmoid_Z * (1 - sigmoid_Z)

    def tanh_derivative(self, Z):
        """ المشتقة الأولى لدالة التانجنت هايبر """
        tanh_Z = self.tanh(Z)
        return 1 - tanh_Z ** 2

    def forward(self, X):
        """ التمرير الأمامي """
        self.z = []  # تخزين القيم قبل التفعيل
        self.output = [X]  # الطبقة الأولى هي المدخلات

        for i in range(len(self.weights)):
            z = np.dot(self.output[-1], self.weights[i].T) + (self.bias[i] if self.isBias else 0)
            self.z.append(z)
            # Apply the correct activation function based on user choice
            if self.isSigmoid:
                self.output.append(self.sigmoid(z))
            else:
                self.output.append(self.tanh(z))

        # طبقة الإخراج
        z_out = np.dot(self.output[-1], self.weights_output.T) + (self.bias_output if self.isBias else 0)
        self.z.append(z_out)
        # Apply the correct activation function for the output layer
        if self.isSigmoid:
            final_output = self.sigmoid(z_out)
        else:
            final_output = self.tanh(z_out)

        self.output.append(final_output)

        return final_output

    def backward(self, X, Y):
        """ التمرير العكسي لحساب المشتقات وتحديث الأوزان """
        m = X.shape[0]  # عدد العينات
        gradients = {}

        # خطأ طبقة الإخراج
        dz = self.output[-1] - Y
        gradients["dw_output"] = (1 / m) * np.dot(dz.T, self.output[-2])

        dz_array = dz.to_numpy()  # تحويل البيانات لـ NumPy array
        gradients["db_output"] = (1 / m) * np.sum(dz_array, axis=0, keepdims=True) if self.isBias else 0

        # العودة عبر الطبقات المخفية
        for i in reversed(range(len(self.weights))):
            da = np.dot(dz, self.weights_output if i == len(self.weights) - 1 else self.weights[i + 1])
            if self.isSigmoid:
                dz = da * self.sigmoid_derivative(self.z[i])
            else:
                dz = da * self.tanh_derivative(self.z[i])
            gradients[f"dw_{i}"] = (1 / m) * np.dot(dz.T, self.output[i])
            gradients[f"db_{i}"] = (1 / m) * np.sum(dz, axis=0, keepdims=True) if self.isBias else 0

        # تحديث الأوزان والانحيازات
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients[f"dw_{i}"]
            if self.isBias:
                self.bias[i] -= self.learning_rate * gradients[f"db_{i}"]

        self.weights_output -= self.learning_rate * gradients["dw_output"]
        if self.isBias:
            self.bias_output -= self.learning_rate * gradients["db_output"]

    def train(self, X, Y):
        """ تدريب الشبكة العصبية """
        self.get_sizes(X, Y)
        self.initialize_params(X, Y)

        for epoch in range(self.epochs):
            predictions = self.forward(X)
            cost = np.mean((predictions - Y) ** 2)

            self.backward(X, Y)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch + 1}/{self.epochs}, Cost: {cost}")


# بيانات تجريبية
if __name__ == "__main__":
    X = np.random.rand(100, 5)  # 100 عينات بـ 5 ميزات
    Y = np.random.rand(100, 3)  # 3 مخرجات

    nn = BackPropagation(learning_rate=0.01, layers=2, epochs=100, neurons=[4, 3], isBias=True)
    nn.train(X, Y)

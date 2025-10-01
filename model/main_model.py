import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim=2304, num_classes=394, hidden_dims=[1024, 512, 256, 128], dropout_rate=0.3):
        super(Classifier, self).__init__()

        # Список слоев
        layers = []
        in_dim = input_dim

        # Создаем слои динамически на основе hidden_dims
        for out_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),  # Добавляем батч-нормализацию
                nn.GELU(),  # Активация GELU
                nn.Dropout(dropout_rate)  # Dropout для регуляризации
            ])
            in_dim = out_dim

        # Финальный слой
        layers.append(nn.Linear(in_dim, num_classes))

        # Собираем сеть (без Softmax в __init__, так как он будет в forward)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Предполагаем, что x имеет размер (bs, n, b)
        x = x.reshape(x.size(0), -1)  # Динамическое преобразование в (bs, n*b)
        logits = self.net(x)  # Получаем логиты
        return nn.functional.softmax(logits, dim=1)  # Применяем Softmax

    def get_logits(self, x):
        # Метод для получения логитов (без Softmax, для кросс-энтропии)
        x = x.reshape(x.size(0), -1)
        return self.net(x)
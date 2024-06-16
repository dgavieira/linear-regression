
# Regressão Linear com Adaline

Este é um projeto de implementação de regressão linear utilizando o algoritmo Adaline. O objetivo é prever a pressão sanguínea máxima de pacientes com base em variáveis como sexo, idade, peso e se são fumantes ou não.
Este código trata-se da implementação do segundo trabalho da disciplina Aprendizado de Máquina do curso de mestrado em Engenharia Elétrica pelo Programa de Pós-Graduação em Engenharia Elétrica da Universidade Federal do Amazonas.

* Tema: Regressão Linear com Adaline
* Disciplina: PGENE 556 - Aprendizado de Máquina
* PPGEE - Programa de Pós Graduação em Engenharia Elétrica
* UFAM - Universidade Federal do Amazonas
* Autor: Diego Giovanni de ALcântara Vieira.

## Instalação

1. Clone este repositório:
```
git clone https://github.com/dgavieira/linear-regression.git
```

1. Instale as dependências:

```
python -m venv .venv
source .venv/bin/activate
python setup.py install
```

## Uso

1. Execute o arquivo `main.py` para treinar o modelo e gerar os resultados.
```
python src/main.py
```

1. Os resultados serão salvos em um arquivo `results.csv` e os gráficos em uma pasta `images`.

## Estrutura do Projeto

- `data/`: Contém a base de dados `hospital.xls`.
- `models/`: Contém as implementações do `Adaline` e `StandardScaler`.
- `src/`: Contém o código principal, incluindo `main.py` e outros módulos.
- `images/`: Pasta onde os gráficos são salvos.
- `notebooks/`: Representa o baseline do projeto. Implementação do propósito do trabalho mas com funções de auto nível para validar a implementação real contida em `src`.

## Contribuição

Contribuições são bem-vindas! Para sugestões, abra uma issue. Para mudanças significativas, abra um pull request.

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para obter mais informações.
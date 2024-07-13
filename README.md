# calibtools

标定工具箱

## 开发记录

### poetry

#### 安装poetry

```shell
curl -sSL https://install.python-poetry.org | python3 -
poetry --version
```

#### 初始化

```shell
conda create -n calib python=3.8
poetry init
```

#### 切换虚拟环境

```shell
conda activate other_env_python
poetry env info
```

#### 创建新的虚拟环境(optional)

```shell
conda create -n new_env python=xx.xx
poetry install
```

自动版本控制

[Poetry: Automatically generate package version from git commit](https://sam.hooke.me/note/2023/08/poetry-automatically-generated-package-version-from-git-commit/)

### setuptools

[Configuring setuptools using pyproject.toml files](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)

### setuptools_scm

自动版本控制

[setuptools_scm](https://pypi.org/project/setuptools-scm/7.0.3/)

## 安装使用

## contributors

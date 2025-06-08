from setuptools import setup, find_packages

setup(
    name='your_project_name',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'tensorflow',
        'xgboost',
        'statsmodels',
        'prophet'
        # Thêm các thư viện bạn cần ở đây
    ],
    entry_points={
        'console_scripts': [
            'run_app=app:main',  # giả sử bạn định nghĩa hàm main() trong app.py để chạy Streamlit
        ],
    },
)

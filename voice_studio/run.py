import os

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import uvicorn


def main():
    uvicorn.run('voice_studio.server:APP', host='127.0.0.1', port=7860, workers=1, log_level='info')


if __name__ == '__main__':
    main()

from celery import Celery
import config as cfg
def make_celery(app_name=__name__):
    celery_app = Celery(
        app_name,
        broker=cfg.REDIS_URL,
        backend=cfg.REDIS_URL,
        include=['tasks']
    )
    celery_app.conf.update(
        task_track_started=True,
    )
    celery_app.conf.worker_prefetch_multiplier = 1
    return celery_app
    # return Celery(
    #     app_name,
    #    broker=cfg.REDIS_URL,
    #     backend=cfg.REDIS_URL,
    #     include=['tasks']
    # )

celery = make_celery()

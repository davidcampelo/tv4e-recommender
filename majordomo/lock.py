import redis

from django.db import models

from django.conf import settings

lock_table = redis.StrictRedis.from_url(settings.REDIS_URL)


class ConcurrentModificationError(ValueError):
    """Base error class for write concurrency errors"""
    pass


class StaleWriteError(ConcurrentModificationError):
    """Tried to write a version of a model that is older than the current version in the database"""
    pass


class AlreadyLockedError(ConcurrentModificationError):
    """Tried to aquire a lock on a row that is already locked"""
    pass


class WriteWithoutLockError(ConcurrentModificationError):
    """Tried to save a lock-required model row without locking it first"""
    pass


class LockedModel:
    """Add row-level locking backed by redis, set lock_required=True to require a lock on .save()"""

    lock_required = False  # whether a lock is required to call .save() on this model

    @property
    def _lock_key(self):
        model_name = self.__class__.__name__
        return '{0}__locked:{1}'.format(model_name, self.id)

    def is_locked(self):
        return lock_table.get(self._lock_key) == b'1'

    def lock(self):
        if self.is_locked():
            raise AlreadyLockedError('This item is locked right now, please try again later.')
        lock_table.set(self._lock_key, b'1')

    def unlock(self):
        lock_table.set(self._lock_key, b'0')

    def save(self, *args, **kwargs):
        if self.lock_required and not self.is_locked():
            raise WriteWithoutLockError('Tried to save a lock-required model row without locking it first')
        super(LockedModel, self).save(*args, **kwargs)


# example usage to require locking on a model when calling .save():
# class Game(models.Model, LockedModel):
#     lock_required = True

#     players = models.ManyToManyField(Player)

# from django.db import IntegrityError, transaction
# from .models import Game, Player

# def perform_game_action(game: Game, new_player: Player):
#     # acquire redis write-lock on db objects
#     game.lock()
#     try:
#         with transaction.atomic():
#             # modify your database object here
#             game.players.add(new_player)
#             # save all modified state to database
#             game.save()
#     except ConcurrentModificationError, IntegrityError:
#         # handle write integrity errors/lock contention cases here
#         print('Game transaction failed!')
#     finally:
#         # release redis write-lock on table object
#         game.unlock()
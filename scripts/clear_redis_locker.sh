redis-cli KEYS *lock* | xargs redis-cli DEL

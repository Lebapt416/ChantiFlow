-- Migration pour ajouter les champs de planification sur les tâches
ALTER TABLE tasks
ADD COLUMN IF NOT EXISTS planned_start TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS planned_end TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS planned_order INTEGER,
ADD COLUMN IF NOT EXISTS planned_worker_id UUID REFERENCES workers(id);



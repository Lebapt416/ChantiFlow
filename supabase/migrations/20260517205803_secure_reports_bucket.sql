-- Limiter le bucket reports : taille max 10MB, mime types image seulement
UPDATE storage.buckets
SET
  file_size_limit = 10485760,  -- 10 MB en bytes
  allowed_mime_types = ARRAY['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif']
WHERE name = 'reports';

-- RLS sur storage.objects : seuls les workers authentifiés peuvent uploader
-- (la lecture reste publique car le bucket est marqué public)
CREATE POLICY IF NOT EXISTS "Authenticated workers can upload to reports"
  ON storage.objects FOR INSERT
  WITH CHECK (
    bucket_id = 'reports'
    AND auth.role() IN ('authenticated', 'service_role')
  );

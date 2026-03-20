// ---- Piattaforma X-HEEP / SPI flash (mantieni/porta secondo target) ----
#include "x-heep.h"
#include "w25q128jw.h"
#include "csr.h"
#include "build_config.h"

// =================== Helpers FLASH ===================
uint32_t *heep_get_flash_address_offset(uint32_t *data_address_lma);
static inline w25q_error_codes_t fill_buffer(uint32_t *source, uint32_t *buffer, uint32_t len_words)
{
  uint32_t *source_flash = heep_get_flash_address_offset(source);
#if FLASH_USE_QUAD
  return w25q128jw_read_quad((uint32_t)source_flash, buffer, len_words * 4);
#else
  return w25q128jw_read_standard((uint32_t)source_flash, buffer, len_words * 4);
#endif
}
static inline w25q_error_codes_t flash_read_bytes(const void *src_sym, void *dst_buf, size_t nbytes)
{
  size_t words = (nbytes + 3u) / 4u;
  return fill_buffer((uint32_t *)src_sym, (uint32_t *)dst_buf, (uint32_t)words);
}
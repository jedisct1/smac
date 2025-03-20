#define smac_MACBYTES   32
#define smac_KEYBYTES   32
#define smac_NONCEBYTES 16
#define smac_BLOCKBYTES 16

#include <stddef.h>
#include <string.h>

#ifdef __x86_64__
#    include <immintrin.h>
#    include <wmmintrin.h>

typedef __m128i aes_block_t;

#    define AES_BLOCK_XOR(A, B)       _mm_xor_si128((A), (B))
#    define AES_BLOCK_LOAD(A)         _mm_loadu_si128((const aes_block_t *) (const void *) (A))
#    define AES_BLOCK_LOAD_64x2(A, B) _mm_set_epi64x((long long) (A), (long long) (B))
#    define AES_BLOCK_STORE(A, B)     _mm_storeu_si128((aes_block_t *) (void *) (A), (B))
#    define AES_ENC(A, B)             _mm_aesenc_si128((A), (B))
#    define PERMUTE(A) \
        _mm_shuffle_epi8((A), _mm_setr_epi8(0, 7, 14, 11, 4, 13, 10, 1, 8, 15, 6, 3, 12, 5, 2, 9))
#elif defined __aarch64__
#    include <arm_neon.h>

typedef uint8x16_t aes_block_t;

#    define AES_BLOCK_XOR(A, B) veorq_u8((A), (B))
#    define AES_BLOCK_LOAD(A)   vld1q_u8((const uint8_t *) (A))
#    define AES_BLOCK_LOAD_64x2(A, B) \
        vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(B), vcreate_u64(A)))
#    define AES_BLOCK_STORE(A, B) vst1q_u8((uint8_t *) (A), (B))
#    define AES_ENC(A, B)         veorq_u8(vaesmcq_u8(vaeseq_u8((A), vmovq_n_u8(0))), (B))
#    define PERMUTE(A)              \
        vqtbl1q_u8(                 \
            (A),                    \
            vld1q_u8((const uint8_t \
                          *) "\x00\x07\x0e\x0b\x04\x0d\x0a\x01\x08\x0f\x06\x03\x0c\x05\x02\x09"))
#else
#    error "Unsupported architecture"
#endif

#define COMPRESS(state, m)                                                                         \
    do {                                                                                           \
        state = (smac_state) { .a1 = PERMUTE(AES_BLOCK_XOR(AES_BLOCK_XOR(state.a2, state.a3), m)), \
                               .a2 = AES_ENC(state.a1, m),                                         \
                               .a3 = AES_ENC(state.a2, m) };                                       \
    } while (0)

typedef struct smac_state {
    aes_block_t a1, a2, a3;
} smac_state;

void
smac(uint8_t tag[smac_MACBYTES], const uint8_t *ad, const size_t ad_len, const uint8_t *ct,
     const size_t ct_len, const uint8_t key[smac_KEYBYTES], const uint8_t nonce[smac_NONCEBYTES])
{
    smac_state  state, s0;
    uint8_t     pad[smac_BLOCKBYTES];
    aes_block_t m;
    size_t      i, left;

    state = (smac_state) { .a1 = AES_BLOCK_LOAD(key),
                           .a2 = AES_BLOCK_LOAD(key + 16),
                           .a3 = AES_BLOCK_LOAD(nonce) };
    s0    = state;
    m     = AES_BLOCK_LOAD_64x2(0, 1);
    for (i = 0; i < 9; i++) {
        COMPRESS(state, m);
    }
    state = (smac_state) { .a1 = AES_BLOCK_XOR(state.a1, s0.a1),
                           .a2 = AES_BLOCK_XOR(state.a2, s0.a2),
                           .a3 = AES_BLOCK_XOR(state.a3, s0.a3) };

    for (i = 0; i + smac_BLOCKBYTES * 2 <= ad_len; i += smac_BLOCKBYTES * 2) {
        m = AES_BLOCK_LOAD(ad + i);
        COMPRESS(state, m);
        m = AES_BLOCK_LOAD(ad + i + smac_BLOCKBYTES);
        COMPRESS(state, m);
    }
    for (; i + smac_BLOCKBYTES <= ad_len; i += smac_BLOCKBYTES) {
        m = AES_BLOCK_LOAD(ad + i);
        COMPRESS(state, m);
    }
    left = ad_len % smac_BLOCKBYTES;
    if (left != 0) {
        memset(pad, 0, sizeof pad);
        memcpy(pad, ad, left);
        m = AES_BLOCK_LOAD(pad);
        COMPRESS(state, m);
    }

    for (i = 0; i + smac_BLOCKBYTES * 2 <= ct_len; i += smac_BLOCKBYTES * 2) {
        m = AES_BLOCK_LOAD(ct + i);
        COMPRESS(state, m);
        m = AES_BLOCK_LOAD(ct + i + smac_BLOCKBYTES);
        COMPRESS(state, m);
    }
    for (; i + smac_BLOCKBYTES <= ct_len; i += smac_BLOCKBYTES) {
        m = AES_BLOCK_LOAD(ct + i);
        COMPRESS(state, m);
    }
    left = ct_len % smac_BLOCKBYTES;
    if (left != 0) {
        memset(pad, 0, sizeof pad);
        memcpy(pad, ct, left);
        m = AES_BLOCK_LOAD(pad);
        COMPRESS(state, m);
    }

    m = AES_BLOCK_LOAD_64x2(ct_len * 8, ad_len * 8);
    COMPRESS(state, m);

    s0 = state;
    m  = AES_BLOCK_LOAD_64x2(0, 1);
    for (i = 0; i < 9; i++) {
        COMPRESS(state, m);
    }

    state = (smac_state) { .a1 = AES_BLOCK_XOR(state.a1, s0.a1),
                           .a2 = AES_BLOCK_XOR(state.a2, s0.a2),
                           .a3 = AES_BLOCK_XOR(state.a3, s0.a3) };

    AES_BLOCK_STORE(tag, state.a2);
    AES_BLOCK_STORE(tag + 16, state.a3);
}

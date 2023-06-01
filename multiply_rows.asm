.CODE

; multiplyRows(double *Q, double *U, int k, int i, int j, int n)
; temp = 0;
; for (int l = k; l < n; ++l) {
;    temp += Q[l][j] * U[k][l];
; }
; temp *= Q[i][k];

; Q = rcx
; U = rdx
; k = r8
; i = r9
; j = rsp+56
; n = rsp+64

multiplyRows PROC frame

    push rsi
    .pushreg rsi
    push rdi
    .pushreg rdi
    .endprolog

    mov rsi, rcx; rsi = Q
    mov rdi, rdx; rdi = U

    vxorpd xmm0, xmm0, xmm0; temp = 0
    mov eax, r8d; l = k

    @@:
        mov ebx, [rsp+56]; ebx = j
        mov edx, [rsp+64]; ebx = n

        mov ecx, eax; ecx = l
        imul ecx, edx; ecx = l*n
        add ecx, ebx; ecx = l*n+j
        vmovsd xmm2, real8 ptr [rsi+rcx*8]; xmm2 = Q[l][j]

        mov ecx, r8d; rcx = k
        imul ecx, edx; rcx = k*n
        add ecx, eax; rcx = k*n+l
        vmovsd xmm3, real8 ptr [rdi+rcx*8]; xmm3 = U[k][l]

        vmulsd xmm1, xmm2, xmm3; xmm1 = Q[l][j] * U[k][l]
        vaddsd xmm0, xmm0, xmm1; temp += xmm1

        inc eax; ++l
        cmp eax, edx; if (l<n)
    jl @B

    mov ecx, r9d; rcx = i
    imul ecx, edx; rcx = i*n
    add ecx, r8d; rcx = i*n+k
    vmovsd xmm4, real8 ptr [rsi+rcx*8]; xmm4 = Q[i][k]

    vmulsd xmm0, xmm0, xmm4; temp *= Q[i][k]

    pop rdi
    pop rsi

    ret

multiplyRows ENDP

END
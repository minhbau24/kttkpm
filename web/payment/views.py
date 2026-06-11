import json
import random
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from orders.models import Order
from shipping.models import Shipment
from .models import Payment

@csrf_exempt
@login_required
def payment_pay_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except:
            data = request.POST

        order_id = data.get('order_id')
        if not order_id:
            return JsonResponse({'error': 'order_id is required'}, status=400)

        payment = get_object_or_404(Payment, order_id=order_id)
        if payment.status == 'success':
            return JsonResponse({'message': 'Order is already paid'}, status=400)

        order = get_object_or_404(Order, id=order_id, user=request.user)

        success = random.random() < 0.95
        if success:
            payment.status = 'success'
            payment.transaction_id = f"TXN-{random.randint(10000000, 99999999)}"
            payment.save()

            order.status = 'paid'
            order.save()

            address = request.session.get(f'shipping_address_{order.id}', "Địa chỉ giao hàng mặc định")
            
            # Trigger shipping microservice
            Shipment.objects.create(
                order_id=order.id,
                address=address,
                status='processing',
                tracking_number=f"TRK-{random.randint(10000000, 99999999)}"
            )

            return JsonResponse({
                'message': 'Payment successful',
                'payment_status': payment.status,
                'transaction_id': payment.transaction_id,
                'order_status': order.status
            })
        else:
            payment.status = 'failed'
            payment.save()
            return JsonResponse({
                'message': 'Payment failed',
                'payment_status': payment.status
            }, status=400)

    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def payment_status_api(request):
    order_id = request.GET.get('order_id')
    if not order_id:
        return JsonResponse({'error': 'order_id is required'}, status=400)
    payment = get_object_or_404(Payment, order_id=order_id)
    return JsonResponse({
        'order_id': payment.order_id,
        'amount': float(payment.amount),
        'status': payment.status,
        'transaction_id': payment.transaction_id
    })

# HTML views for Storefront mock payment page
@login_required
def order_payment_page(request, order_id):
    order = get_object_or_404(Order, id=order_id, user=request.user)
    payment = get_object_or_404(Payment, order_id=order_id)
    
    if order.status != 'pending':
        messages.info(request, "Đơn hàng đã được thanh toán hoặc xử lý!")
        return redirect('order_detail', pk=order.id)

    if request.method == 'POST':
        action = request.POST.get('action', '')
        if action == 'success':
            payment.status = 'success'
            payment.transaction_id = f"TXN-{random.randint(10000000, 99999999)}"
            payment.save()

            order.status = 'paid'
            order.save()

            address = request.session.get(f'shipping_address_{order.id}', "Số 1 Đại Cồ Việt, Bách Khoa, Hai Bà Trưng, Hà Nội")
            Shipment.objects.get_or_create(
                order_id=order.id,
                defaults={
                    'address': address,
                    'status': 'processing',
                    'tracking_number': f"TRK-{random.randint(10000000, 99999999)}"
                }
            )

            messages.success(request, "Thanh toán thành công! Đơn hàng đang được chuẩn bị vận chuyển.")
            return redirect('order_detail', pk=order.id)
        else:
            payment.status = 'failed'
            payment.save()
            messages.error(request, "Thanh toán thất bại! Vui lòng thử lại.")
            return redirect('order_detail', pk=order.id)

    return render(request, 'payment/payment_page.html', {'order': order, 'payment': payment})

import json
import random
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from orders.models import Order
from .models import Shipment

@csrf_exempt
@login_required
def shipping_create_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except:
            data = request.POST

        order_id = data.get('order_id')
        address = data.get('address')

        if not order_id or not address:
            return JsonResponse({'error': 'order_id and address are required'}, status=400)

        if Shipment.objects.filter(order_id=order_id).exists():
            return JsonResponse({'error': 'Shipment already exists for this order'}, status=400)

        shipment = Shipment.objects.create(
            order_id=order_id,
            address=address,
            status='processing',
            tracking_number=f"TRK-{random.randint(10000000, 99999999)}"
        )

        return JsonResponse({
            'message': 'Shipment created successfully',
            'shipment_id': shipment.id,
            'tracking_number': shipment.tracking_number,
            'status': shipment.status
        }, status=201)

    return JsonResponse({'error': 'Method not allowed'}, status=405)

@login_required
def shipping_status_api(request):
    order_id = request.GET.get('order_id')
    if not order_id:
        return JsonResponse({'error': 'order_id is required'}, status=400)

    shipment = get_object_or_404(Shipment, order_id=order_id)
    return JsonResponse({
        'order_id': shipment.order_id,
        'tracking_number': shipment.tracking_number,
        'address': shipment.address,
        'status': shipment.status
    })

# HTML views for Admin or Staff to update shipment status
@login_required
def update_shipping_status(request, order_id):
    if request.user.role not in ['admin', 'staff']:
        messages.error(request, "Bạn không có quyền thực hiện hành động này!")
        return redirect('order_detail', pk=order_id)

    order = get_object_or_404(Order, id=order_id)
    shipment = get_object_or_404(Shipment, order_id=order_id)

    if request.method == 'POST':
        new_status = request.POST.get('status', '')
        if new_status in ['processing', 'shipping', 'delivered']:
            shipment.status = new_status
            shipment.save()

            if new_status == 'delivered':
                order.status = 'completed'
                order.save()
            elif new_status == 'shipping':
                order.status = 'shipping'
                order.save()

            messages.success(request, f"Đã cập nhật trạng thái giao hàng thành: {shipment.get_status_display()}")
        else:
            messages.error(request, "Trạng thái không hợp lệ!")
            
        return redirect('order_detail', pk=order_id)

    return render(request, 'shipping/update_shipping.html', {'order': order, 'shipment': shipment})

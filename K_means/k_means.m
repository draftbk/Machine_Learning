function [ value,center] = k_means(k,data)
[col,row] = size(data)
range = [];

% ȡ���е������k��
point = [];
num=[];
num=fix(rand(1,k)*col);
% �����������Ҫתһ��
 point=data(num,:)';

err = 250;
while(err>0.01)
    result = ones(col,1);
    distance = ones(col,1).*3141592612;

    for i=1:1:k
        cls = point(:,i);
        a = [];
        % ����ŷʽ����
        for j=1:1:row

            temp = data(:,j)-cls(j);
            a = [a,temp.^2];
        end
        temp = sum(a,2);
        result = max([result,(temp<distance).*i]')';
        distance = min([distance,temp]')';
    end
    
    answer = [];
    for i=1:1:k
        temp = (result==i);
%         a�����ĵ�
        a = [];
%         ����
        sumTemp = sum(temp);
        if(sumTemp==0)
            sumTemp = 1;
        end
        for j=1:1:row
            a = [a;(temp')*data(:,j)/sumTemp];
        end
        answer = [answer,a];
    end
%     �������
    err = answer-point;
%     a = error.^2;
    a=[];
    for i = 1:1:row
      a=[a;err(i,:).^2];
    end
    err= max(sum(a));
    point = answer;
end
value = result;
center = answer;
end

